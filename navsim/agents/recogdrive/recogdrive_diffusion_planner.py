# Copyright 2025 The Xiaomi Corporation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import lzma
import math
import pickle
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np
import torch
from timm.models.layers import Mlp
from torch import nn
from torch.distributions import Beta, Normal, kl_divergence
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
    PDMScorerConfig,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)

from .blocks.encoder import (
    ActionEncoder,
    SinusoidalPositionalEncoding,
    StateAttentionEncoder,
    SwiGLUFFN,
)
from .recogdrive_dit import LightningDiT
from .utils.internvl_preprocess import load_image

@dataclass
class FlowConfig:
    """Configuration specific to Flow Matching."""
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000
    mean_variance_net: bool = False

@dataclass
class DDPMConfig:
    """Configuration specific to DDPM."""
    num_train_timesteps: int = 100

@dataclass
class DDIMConfig:
    """Configuration specific to DDIM."""
    num_train_timesteps: int = 100
    ddim_eta: float = 0.0

@dataclass
class GRPOConfig:
    """Configuration specific to GRPO training."""
    denoised_clip_value: float = 1.0
    eval_randn_clip_value: float = 1.0
    randn_clip_value: float = 5.0
    final_action_clip_value: float = 1.0
    eps_clip_value: Optional[float] = None
    eval_min_sampling_denoising_std: float = 0.0001
    min_sampling_denoising_std: float = 0.04
    min_logprob_denoising_std: float = 0.1
    clip_advantage_lower_quantile: float = 0.0
    clip_advantage_upper_quantile: float = 1.0
    gamma_denoising: float = 0.6
    
    metric_cache_path: str = "/path/to/metric_cache_train"
    reference_policy_checkpoint: str = "/path/to/IL_Model.ckpt"
    scorer_config: PDMScorerConfig = field(default_factory=lambda: PDMScorerConfig(
        progress_weight=10.0, ttc_weight=5.0, comfortable_weight=2.0
    ))
    # ELF-VLA Failure-Guided Refinement
    failure_threshold: float = 0.8
    num_refined_samples: int = 4


# @dataclass
# class ReCogDriveDiffusionPlannerConfig(PretrainedConfig):
#     """A refined configuration for the ReCogDriveDiffusionPlanner."""
#     # --- Core Architecture ---
#     diffusion_model_cfg: dict = field(default_factory=dict)
#     input_embedding_dim: int = 1536
#     hidden_size: int = 1024
#     action_dim: int = 3
#     action_horizon: int = 8
#     add_pos_embed: bool = True
#     max_seq_len: int = 8
#     ego_status_encoder_type: Literal['mlp', 'attention'] = 'mlp'

#     sampling_method: Literal['flow', 'ddpm', 'ddim'] = 'ddim'
#     num_inference_steps: int = 5
#     model_dtype: str = "float16"
#     grpo: bool = False
#     vlm_size: str = 'large'
    
#     tune_projector: bool = True
#     tune_diffusion_model: bool = True
    
#     flow_cfg: FlowConfig = field(default_factory=FlowConfig)
#     ddpm_cfg: DDPMConfig = field(default_factory=DDPMConfig)
#     ddim_cfg: DDIMConfig = field(default_factory=DDIMConfig)
#     grpo_cfg: GRPOConfig = field(default_factory=GRPOConfig)
class ReCogDriveDiffusionPlannerConfig(PretrainedConfig):
    def __init__(
        self,
        diffusion_model_cfg=None,
        input_embedding_dim=1536,
        hidden_size=1024,
        action_dim=3,
        action_horizon=8,
        add_pos_embed=True,
        max_seq_len=8,
        ego_status_encoder_type='mlp',
        sampling_method='ddim',
        num_inference_steps=5,
        model_dtype="float16",
        grpo=False,
        vlm_size='large',
        tune_projector=True,
        tune_diffusion_model=True,
        flow_cfg=None,
        ddpm_cfg=None,
        ddim_cfg=None,
        grpo_cfg=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.diffusion_model_cfg = diffusion_model_cfg or {}
        self.input_embedding_dim = input_embedding_dim
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.add_pos_embed = add_pos_embed
        self.max_seq_len = max_seq_len
        self.ego_status_encoder_type = ego_status_encoder_type

        self.sampling_method = sampling_method
        self.num_inference_steps = num_inference_steps
        self.model_dtype = model_dtype
        self.grpo = grpo
        self.vlm_size = vlm_size

        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model

        self.flow_cfg = flow_cfg or FlowConfig()
        self.ddpm_cfg = ddpm_cfg or DDPMConfig()
        self.ddim_cfg = ddim_cfg or DDIMConfig()
        self.grpo_cfg = grpo_cfg or GRPOConfig()


class ReCogDriveDiffusionPlanner(nn.Module):
    config_class = ReCogDriveDiffusionPlannerConfig

    def __init__(self, config: ReCogDriveDiffusionPlannerConfig):
        super().__init__()
        self.config = config
        
        self.model = LightningDiT(**config.diffusion_model_cfg)

        self.his_traj_encoder = Mlp(
            in_features=12,
            hidden_features=config.hidden_size,
            out_features=config.input_embedding_dim,
            norm_layer=nn.LayerNorm
        )

        if config.ego_status_encoder_type == 'attention':
            self.ego_status_encoder = StateAttentionEncoder(
                state_dim=8,
                embed_dim=config.input_embedding_dim,
                num_kinematic_states=4 
            )
        else: 
            self.ego_status_encoder = Mlp(
                in_features=8,
                hidden_features=config.hidden_size,
                out_features=config.input_embedding_dim,
                norm_layer=nn.LayerNorm
            )

        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            hidden_size=config.input_embedding_dim,
        )
        if config.vlm_size == "large":
            self.feature_encoder = nn.Linear(3584, config.input_embedding_dim)
        else:
            self.feature_encoder = nn.Linear(1536, config.input_embedding_dim)
            
        self.fusion_projector = nn.Linear(config.input_embedding_dim * 3, config.input_embedding_dim)

        output_dim = 2 * config.action_dim if (
            config.sampling_method == 'flow' and config.flow_cfg.mean_variance_net
        ) else config.action_dim
        
        self.action_decoder = Mlp(
            in_features=self.model.output_dim,
            hidden_features=config.hidden_size,
            out_features=output_dim,
            norm_layer=nn.LayerNorm
        )
        
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, config.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        
        if self.config.sampling_method == 'flow':
            self._init_flow_sampler(config.flow_cfg)
        elif self.config.sampling_method == 'ddpm':
            self._init_ddpm_sampler(config.ddpm_cfg)
        elif self.config.sampling_method == 'ddim':
            self._init_ddim_sampler(config.ddim_cfg)

        if config.grpo:
            self._init_grpo(config.grpo_cfg)

    def _init_flow_sampler(self, cfg: FlowConfig):
        """Initializes components required for Flow Matching."""
        self.beta_dist = Beta(cfg.noise_beta_alpha, cfg.noise_beta_beta)
        self.num_timestep_buckets = cfg.num_timestep_buckets

    def _init_ddpm_sampler(self, cfg: DDPMConfig):
        """Initializes buffers required for DDPM, using original naming."""
        ddpm_betas = self.cosine_beta_schedule(cfg.num_train_timesteps)
        self.register_buffer('ddpm_betas', ddpm_betas)

        ddpm_alphas = 1.0 - ddpm_betas
        self.register_buffer('ddpm_alphas', ddpm_alphas)

        ddpm_alphas_cumprod = torch.cumprod(ddpm_alphas, dim=0)
        self.register_buffer('ddpm_alphas_cumprod', ddpm_alphas_cumprod)

        ddpm_alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), ddpm_alphas_cumprod[:-1]])
        self.register_buffer('ddpm_alphas_cumprod_prev', ddpm_alphas_cumprod_prev)

        self.register_buffer('ddpm_sqrt_alphas_cumprod', torch.sqrt(ddpm_alphas_cumprod))
        self.register_buffer('ddpm_sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - ddpm_alphas_cumprod))
        self.register_buffer('ddpm_sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / ddpm_alphas_cumprod))
        self.register_buffer('ddpm_sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / ddpm_alphas_cumprod - 1.0))

        ddpm_var = ddpm_betas * (1.0 - ddpm_alphas_cumprod_prev) / (1.0 - ddpm_alphas_cumprod)
        self.register_buffer('ddpm_var', ddpm_var)
        self.register_buffer('ddpm_logvar_clipped', torch.log(ddpm_var.clamp(min=1e-20)))
        
        self.register_buffer('ddpm_mu_coef1', ddpm_betas * torch.sqrt(ddpm_alphas_cumprod_prev) / (1.0 - ddpm_alphas_cumprod))
        self.register_buffer('ddpm_mu_coef2', (1.0 - ddpm_alphas_cumprod_prev) * torch.sqrt(ddpm_alphas) / (1.0 - ddpm_alphas_cumprod))

    def _init_ddim_sampler(self, cfg: DDIMConfig):
        """Initializes buffers required for DDIM sampling, using original naming."""
        self._init_ddpm_sampler(DDPMConfig(num_train_timesteps=cfg.num_train_timesteps))

        self.eta = EtaFixed(base_eta=1.0).to(self.device)
        for param in self.eta.parameters():
            param.requires_grad = False
        ddim_steps = self.config.num_inference_steps
        self.ddim_steps = ddim_steps
        self.ft_denoising_steps = ddim_steps
        ddim_eta = cfg.ddim_eta 

        self.ddpm_num_train_timesteps = cfg.num_train_timesteps
        step_ratio = self.ddpm_num_train_timesteps // ddim_steps
        ddim_t = torch.arange(0, ddim_steps) * step_ratio
        self.register_buffer('ddim_t_schedule', ddim_t.long()) # Die originalen Zeitpunkte

        ddim_alphas = self.ddpm_alphas_cumprod[self.ddim_t_schedule].clone().to(torch.float32)
        ddim_alphas_prev = torch.cat([
            torch.tensor([1.0], dtype=torch.float32),
            self.ddpm_alphas_cumprod[self.ddim_t_schedule[:-1]]
        ])
        ddim_sqrt_one_minus_alphas = (1.0 - ddim_alphas) ** 0.5

        ddim_sigmas = ddim_eta * (
            (1 - ddim_alphas_prev) / (1 - ddim_alphas) *
            (1 - ddim_alphas / ddim_alphas_prev)
        )**0.5

        # Flip all for sampling order (T -> 0)
        def flip_buffer(name, tensor):
            self.register_buffer(name, torch.flip(tensor, [0]))

        flip_buffer('ddim_t', self.ddim_t_schedule)
        flip_buffer('ddim_alphas', ddim_alphas)
        flip_buffer('ddim_alphas_sqrt', torch.sqrt(ddim_alphas))
        flip_buffer('ddim_alphas_prev', ddim_alphas_prev)
        flip_buffer('ddim_sqrt_one_minus_alphas', ddim_sqrt_one_minus_alphas)
        flip_buffer('ddim_sigmas', ddim_sigmas)

    def _init_grpo(self, cfg: GRPOConfig):
        """Initializes components and hyperparameters for GRPO training."""
        self.denoised_clip_value = cfg.denoised_clip_value
        self.eval_randn_clip_value = cfg.eval_randn_clip_value
        self.randn_clip_value = cfg.randn_clip_value
        self.final_action_clip_value = cfg.final_action_clip_value
        self.eps_clip_value = cfg.eps_clip_value
        self.eval_min_sampling_denoising_std = cfg.eval_min_sampling_denoising_std
        self.min_sampling_denoising_std = cfg.min_sampling_denoising_std
        self.min_logprob_denoising_std = cfg.min_logprob_denoising_std
        self.clip_advantage_lower_quantile = cfg.clip_advantage_lower_quantile
        self.clip_advantage_upper_quantile = cfg.clip_advantage_upper_quantile
        self.gamma_denoising = cfg.gamma_denoising
        self.failure_threshold   = cfg.failure_threshold
        self.num_refined_samples = cfg.num_refined_samples

        # --- debug / monitoring counters ---
        self._dbg_step           = 0
        self._dbg_teacher_calls  = 0
        self._dbg_latencies: list = []
        self._dbg_rewards_before: list = []
        self._dbg_rewards_after: list  = []

        # Policy Monitor
        self._mon_ema_reward: Optional[float] = None   # EMA over ALL sampled trajectories
        _EMA_DECAY = 0.99
        self._mon_ema_decay: float = _EMA_DECAY

        # Teacher Monitor
        self._mon_teacher_wins: int = 0                # calls where reward_after > reward_before
        self._mon_reward_deltas: list = []             # (reward_after - reward_before) per call

        # Sliding window – last 500 teacher calls
        self._mon_win_deltas: deque = deque(maxlen=500)
        self._mon_win_wins:   deque = deque(maxlen=500)

        # Refinement Filter counters
        self._mon_ref_attempts:   int   = 0
        self._mon_ref_accepted:   int   = 0
        self._mon_ref_rejected:   int   = 0
        self._mon_best_delta_sum: float = 0.0
        # -----------------------------------

        self.metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
        proposal_sampling = TrajectorySampling(time_horizon=4, interval_length=0.1)
        self.simulator = PDMSimulator(proposal_sampling)
        self.train_scorer = PDMScorer(proposal_sampling, cfg.scorer_config)
        
        try:
            state_dict = torch.load(cfg.reference_policy_checkpoint, map_location="cpu")["state_dict"]
            model_dict = self.state_dict()
            filtered_ckpt = {}
            for k, v in state_dict.items():
                if k.startswith("agent.action_head."):
                    k2 = k[len("agent.action_head."):]
                else:
                    k2 = k
                if k2 in model_dict and v.shape == model_dict[k2].shape:
                    filtered_ckpt[k2] = v
                else:
                    print(f"Skip loading '{k}' → '{k2}' (checkpoint shape {tuple(v.shape)} vs model shape {tuple(model_dict.get(k2, v).shape)})")
            self.load_state_dict(filtered_ckpt, strict=True)
        except FileNotFoundError:
            print(f"Warning: GRPO checkpoint not found at {cfg.reference_policy_checkpoint}. Skipping loading.")
        
        self.old_policy = copy.deepcopy(self)
        self.old_policy.eval()
        for param in self.old_policy.parameters():
            param.requires_grad = False

    @staticmethod
    def cosine_beta_schedule(timesteps: int, s: float = 0.008, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Calculates a cosine noise schedule as proposed in the iDDPM paper.
        
        This method is static as it does not depend on the instance's state.
        """
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
        return torch.tensor(betas_clipped, dtype=dtype)

    @staticmethod
    def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """
        Extracts values from tensor `a` at indices `t` and reshapes them
        to be broadcastable with a tensor of shape `x_shape`.
        
        This method is static as it does not depend on the instance's state.
        """
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    @staticmethod
    def make_timesteps(batch_size: int, i: int, device: torch.device) -> torch.Tensor:
        """
        Creates a tensor of a constant value `i` for a given batch size and device.
        
        This method is static as it does not depend on the instance's state.
        """
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)
        return t


    def set_frozen_modules_to_eval_mode(self):
        """
        Sets frozen parts of the model to evaluation mode during training.
        This is necessary to disable behaviors like dropout in the frozen layers.
        """
        if self.training:

            if not self.config.tune_projector:
                self.his_traj_encoder.eval()
                self.ego_status_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                self.feature_encoder.eval()
                self.fusion_projector.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            
            if not self.config.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        """Samples time for training based on the sampling method."""
        if self.config.sampling_method == 'flow':
            sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
            return (self.config.flow_cfg.noise_s - sample) / self.config.flow_cfg.noise_s
        elif self.config.sampling_method in ['ddpm', 'ddim']:
            return torch.randint(0, self.ddpm_num_train_timesteps, (batch_size,), device=device).long()
        else:
            raise ValueError(f"Unsupported sampling method: {self.config.sampling_method}")


    def p_mean_variance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        index: torch.Tensor,
        vl_features: torch.Tensor,
        his_traj_features: torch.Tensor,
        ego_status_features: torch.Tensor,
        deterministic: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the mean and log variance of the reverse process p(x_{t-1} | x_t).
        Also returns the predicted x0.
        """
        model_dtype = next(self.model.parameters()).dtype
        x = x.to(model_dtype)
        action_features = self.action_encoder(x, t)
        if hasattr(self, 'position_embedding'):
            pos_ids = torch.arange(action_features.shape[1], device=x.device)
            action_features = action_features + self.position_embedding(pos_ids)

        vl_features_mean = vl_features.mean(1).unsqueeze(1).repeat(1, self.config.action_horizon, 1)
        fused_input = self.fusion_projector(
            torch.cat((his_traj_features, vl_features_mean, action_features), dim=2)
        )

        model_output = self.model(
            hidden_states=fused_input,
            encoder_hidden_states=vl_features,
            conditioning_features=ego_status_features,
            timesteps=t
        )
        pred_noise = self.action_decoder(model_output)

        if self.config.sampling_method == 'ddpm':
            x_recon = self.extract(self.ddpm_sqrt_recip_alphas_cumprod, t, x.shape) * x - \
                      self.extract(self.ddpm_sqrt_recipm1_alphas_cumprod, t, x.shape) * pred_noise
        elif self.config.sampling_method == 'ddim':
            alpha_t = self.extract(self.ddim_alphas, index, x.shape)
            sqrt_one_minus_alpha_t = self.extract(self.ddim_sqrt_one_minus_alphas, index, x.shape)
            x_recon = (x - sqrt_one_minus_alpha_t * pred_noise) / (alpha_t**0.5)
        else:
             raise ValueError(f"p_mean_variance not supported for method: {self.config.sampling_method}")

        denoised_clip_value = getattr(self, 'denoised_clip_value', 1.0)
        x_recon.clamp_(-denoised_clip_value, denoised_clip_value)

        if self.config.sampling_method == 'ddpm':
            model_mean = self.extract(self.ddpm_mu_coef1, t, x.shape) * x_recon + \
                         self.extract(self.ddpm_mu_coef2, t, x.shape) * x
            model_log_variance = self.extract(self.ddpm_logvar_clipped, t, x.shape)
        elif self.config.sampling_method == 'ddim':
            alpha_prev = self.extract(self.ddim_alphas_prev, index, x.shape)
            
            pred_noise = (x - (alpha_t**0.5) * x_recon) / sqrt_one_minus_alpha_t

            eps_clip_value = getattr(self, 'eps_clip_value', None)
            if eps_clip_value is not None:
                pred_noise.clamp_(-eps_clip_value, eps_clip_value)

            if deterministic:
                etas = torch.zeros((x.shape[0], 1, 1)).to(x.device)
            else:
                etas = self.eta(x).unsqueeze(1)

            sigma = (
                etas
                * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)) ** 0.5
            ).clamp_(min=1e-10)

            pred_dir_xt = (1.0 - alpha_prev - sigma**2).clamp(min=0).sqrt() * pred_noise
            model_mean = (alpha_prev**0.5) * x_recon + pred_dir_xt
            model_log_variance = torch.log(sigma**2 + 1e-20)

        return model_mean, model_log_variance, x_recon

    def forward(self, vl_features: torch.Tensor, action_input: BatchFeature) -> BatchFeature:
        """
        Computes the training loss for a given batch.

        Args:
            vl_features (torch.Tensor): The vision-language features from the backbone.
            action_input (BatchFeature): A batch containing ground truth actions and other conditioning.

        Returns:
            BatchFeature: A batch containing the computed loss.
        """
        
        vl_embeds = self.feature_encoder(vl_features)
        his_traj_features = self.his_traj_encoder(
            action_input.his_traj.unsqueeze(1)
        ).repeat(1, self.config.action_horizon, 1)
        ego_status_features = self.ego_status_encoder(action_input.status_feature)
        
        gt_actions = self.norm_odo(action_input.action)

        if self.config.sampling_method == 'flow':
            noise = torch.randn_like(gt_actions)
            t_cont = self.sample_time(gt_actions.shape[0], device=gt_actions.device, dtype=gt_actions.dtype)
            t_cont_reshaped = t_cont[:, None, None]
            
            noisy_actions = (1 - t_cont_reshaped) * noise + t_cont_reshaped * gt_actions
            velocity_target = gt_actions - noise
            t_discrete = (t_cont * self.num_timestep_buckets).long()

            action_features = self.action_encoder(noisy_actions, t_discrete)
            if hasattr(self, 'position_embedding'):
                pos_ids = torch.arange(action_features.shape[1], device=gt_actions.device)
                action_features += self.position_embedding(pos_ids)
            
            vl_embeds_mean = vl_embeds.mean(1).unsqueeze(1).repeat(1, self.config.action_horizon, 1)
            fused_input = self.fusion_projector(
                torch.cat((his_traj_features, vl_embeds_mean, action_features), dim=2)
            )

            model_output = self.model(fused_input, vl_embeds, ego_status_features, t_discrete)
            pred_velocity = self.action_decoder(model_output)
            loss = F.mse_loss(pred_velocity, velocity_target, reduction='mean')
        else: 
            noise = torch.randn_like(gt_actions)
            t_discrete = self.sample_time(gt_actions.shape[0], device=gt_actions.device, dtype=gt_actions.dtype)
            
            noisy_actions = (
                self.extract(self.ddpm_sqrt_alphas_cumprod, t_discrete, gt_actions.shape) * gt_actions +
                self.extract(self.ddpm_sqrt_one_minus_alphas_cumprod, t_discrete, gt_actions.shape) * noise
            )
            
            action_features = self.action_encoder(noisy_actions, t_discrete)
            if hasattr(self, 'position_embedding'):
                pos_ids = torch.arange(action_features.shape[1], device=gt_actions.device)
                action_features += self.position_embedding(pos_ids)
            
            vl_embeds_mean = vl_embeds.mean(1).unsqueeze(1).repeat(1, self.config.action_horizon, 1)
            fused_input = self.fusion_projector(
                torch.cat((his_traj_features, vl_embeds_mean, action_features), dim=2)
            )
            
            model_output = self.model(fused_input, vl_embeds, ego_status_features, t_discrete)
            pred_noise = self.action_decoder(model_output)
            loss = F.mse_loss(pred_noise, noise, reduction='mean')

        return BatchFeature(data={"loss": loss})

    def get_action(
        self,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
        init_actions: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> BatchFeature:
        """
        Generates action trajectories via the configured sampling method.

        This method strictly preserves the original logic for each sampler,
        including specific clipping and noise handling for DDPM and DDIM.

        Args:
            vl_features (torch.Tensor): Vision-language features from the backbone.
            action_input (BatchFeature): Input containing conditioning features like
                historical trajectory and ego status.
            init_actions (Optional[torch.Tensor]): An initial trajectory to start
                the denoising from. If None, starts from pure noise.
            deterministic (bool): If True, DDIM sampling will be deterministic (eta=0).

        Returns:
            BatchFeature: A batch containing the final predicted trajectory.
        """
        vl_embeds = self.feature_encoder(vl_features)
        
        history_embeds = self.his_traj_encoder(
            action_input.his_traj.unsqueeze(1)
        ).repeat(1, self.config.action_horizon, 1)

        ego_embeds = self.ego_status_encoder(
            action_input.status_feature
        )

        B, D = vl_embeds.shape[0], self.config.action_dim
        device, dtype = vl_embeds.device, vl_embeds.dtype
        
        current_actions = init_actions if init_actions is not None else torch.randn(
            (B, self.config.action_horizon, D), device=device, dtype=dtype
        )

        if self.config.sampling_method == 'flow':
            dt = 1.0 / self.config.num_inference_steps
            for step in range(self.config.num_inference_steps):
                idx = int(step / self.config.num_inference_steps * self.config.flow_cfg.num_timestep_buckets)
                t = torch.full((B,), idx, device=device, dtype=torch.long)

                action_features = self.action_encoder(current_actions, t)
                if hasattr(self, 'position_embedding'):
                    action_features += self.position_embedding(torch.arange(self.config.action_horizon, device=device))
                
                vl_embeds_mean = vl_embeds.mean(1).unsqueeze(1).repeat(1, self.config.action_horizon, 1)
                fused_input = self.fusion_projector(
                    torch.cat((history_embeds, vl_embeds_mean, action_features), dim=2)
                )
                
                model_output = self.model(fused_input, vl_embeds, ego_embeds, t)
                pred = self.action_decoder(model_output)
                
                pred_flow = pred.chunk(2, dim=-1)[0] if self.config.flow_cfg.mean_variance_net else pred
                current_actions = current_actions + dt * pred_flow

        elif self.config.sampling_method == 'ddpm':
            step_size = self.config.ddpm_cfg.num_train_timesteps // self.config.num_inference_steps
            timesteps_to_iterate = list(reversed(range(0, self.config.ddpm_cfg.num_train_timesteps, step_size)))
            
            for i, t_int in enumerate(timesteps_to_iterate):
                t_batch = self.make_timesteps(B, t_int, device)
                index_batch = self.make_timesteps(B, i, device)

                mean, logvar, _ = self.p_mean_variance(
                    current_actions, t_batch, index_batch, vl_embeds, history_embeds, ego_embeds, deterministic
                )

                noise_sample = torch.randn_like(current_actions)
                std = torch.exp(0.5 * logvar)

                std = std.to(dtype)

                if t_int == 0:
                    std.zero_()
                else:
                    std = torch.clamp(std, min=1e-3)

                if hasattr(self, 'eval_randn_clip_value') and self.eval_randn_clip_value is not None:
                    noise_sample.clamp_(-self.eval_randn_clip_value, self.eval_randn_clip_value)

                current_actions = mean + std * noise_sample
                
                if hasattr(self, 'final_action_clip_value') and self.final_action_clip_value is not None and i == len(timesteps_to_iterate) - 1:
                    current_actions.clamp_(-self.final_action_clip_value, self.final_action_clip_value)

        elif self.config.sampling_method == 'ddim':
            eval_min_sampling_denoising_std = getattr(self, 'eval_min_sampling_denoising_std', 0.0001)
            eval_randn_clip_value = getattr(self, 'eval_randn_clip_value', 1.0)
            for i in range(self.ddim_steps):
                t_batch = self.make_timesteps(B, self.ddim_t[i], device)
                index_batch = self.make_timesteps(B, i, device)

                mean, logvar, _ = self.p_mean_variance(
                    current_actions, t_batch, index_batch, vl_embeds, history_embeds, ego_embeds, deterministic
                )

                std = torch.exp(0.5 * logvar)

                std = std.to(dtype)

                noise_sample = torch.randn_like(current_actions)
                
                if deterministic:
                    std.zero_()
                else:
                    std = std.clamp(min=eval_min_sampling_denoising_std)
                
                noise_sample.clamp_(-eval_randn_clip_value, eval_randn_clip_value)
                
                current_actions = mean + std * noise_sample
        else:
            raise ValueError(f"Unsupported sampling method: {self.config.sampling_method}")

        final_action_clip_value = getattr(self, 'final_action_clip_value', 1.0)
        if final_action_clip_value is not None:
            current_actions.clamp_(-final_action_clip_value, final_action_clip_value)

        final_actions = self.denorm_odo(current_actions)

        return BatchFeature(data={"pred_traj": final_actions})

    def sample_chain(
        self,
        vl_features: torch.Tensor,
        his_traj_features: torch.Tensor,
        ego_status_features: torch.Tensor,
        init_actions: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ):
        """
        Generates the full denoising chain and the final trajectory.
        This method reuses the logic from get_action but stores intermediate steps.

        Args:
            vl_features (torch.Tensor): Vision-language features from the backbone.
            his_traj_features (torch.Tensor): Encoded historical trajectory features.
            ego_status_features (torch.Tensor): Encoded ego status features.
            init_actions (Optional[torch.Tensor]): An initial trajectory to start from.
                If None, starts from pure noise.
            deterministic (bool): If True, DDIM sampling will be deterministic.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The full denoising chain as a tensor of shape (B, K+1, H, D).
                - The final, denormalized trajectory of shape (B, H, D).
        """
        B, D = vl_features.shape[0], self.config.action_dim
        device, dtype = vl_features.device, vl_features.dtype
        
        vl_features = self.feature_encoder(vl_features)
        
        his_traj_features = self.his_traj_encoder(
            his_traj_features.unsqueeze(1)               
        ).repeat(1, self.config.action_horizon, 1) 

        ego_status_features = self.ego_status_encoder(
            ego_status_features       
        )

        current_actions = init_actions if init_actions is not None else torch.randn(
            (B, self.config.action_horizon, D), device=device, dtype=dtype
        )
        denoising_chain = [current_actions.clone()]

        if self.config.sampling_method == 'flow':
            dt = 1.0 / self.config.num_inference_steps
            for step in range(self.config.num_inference_steps):
                idx = int(step / self.config.num_inference_steps * self.config.flow_cfg.num_timestep_buckets)
                t_batch = torch.full((B,), idx, device=device, dtype=torch.long)
                
                action_features = self.action_encoder(current_actions, t_batch)
                if hasattr(self, 'position_embedding'):
                    action_features += self.position_embedding(torch.arange(self.config.action_horizon, device=device))
                
                vl_features_mean = vl_features.mean(1).unsqueeze(1).repeat(1, self.config.action_horizon, 1)
                fused_input = self.fusion_projector(
                    torch.cat((his_traj_features, vl_features_mean, action_features), dim=2)
                )
                
                model_output = self.model(fused_input, vl_features, ego_status_features, t_batch)
                pred = self.action_decoder(model_output)
                
                pred_flow = pred.chunk(2, dim=-1)[0] if self.config.flow_cfg.mean_variance_net else pred
                current_actions = current_actions + dt * pred_flow
                denoising_chain.append(current_actions.clone())

        elif self.config.sampling_method in ['ddpm', 'ddim']:
            if self.config.sampling_method == 'ddpm':
                step_size = self.config.ddpm_cfg.num_train_timesteps // self.config.num_inference_steps
                timesteps = list(reversed(range(0, self.config.ddpm_cfg.num_train_timesteps, step_size)))
            else: 
                timesteps = self.ddim_t
            
            for i, t_int in enumerate(timesteps):
                t_batch = self.make_timesteps(B, t_int, device)
                index_batch = self.make_timesteps(B, i, device) if self.config.sampling_method == 'ddim' else t_batch

                mean, logvar, _ = self.p_mean_variance(
                    current_actions, t_batch, index_batch, vl_features, his_traj_features, ego_status_features, deterministic
                )

                std = torch.exp(0.5 * logvar).to(dtype)
                noise_sample = torch.randn_like(current_actions)

                if self.config.sampling_method == 'ddim':
                    if deterministic:
                        std.zero_()
                    else:
                        std = std.clamp(min=self.min_sampling_denoising_std)
                else: # ddpm
                    if deterministic and t_int == 0:
                        std = torch.zeros_like(std)
                    elif deterministic:
                        std = std.clamp(min=1e-3)
                    else:
                        std = std.clamp(min=self.min_sampling_denoising_std)
                
                if hasattr(self, 'randn_clip_value') and self.randn_clip_value is not None:
                    noise_sample = noise_sample.clamp_(-self.randn_clip_value, self.randn_clip_value)

                current_actions = mean + std * noise_sample
                
                if i == len(timesteps) - 1 and hasattr(self, 'final_action_clip_value') and self.final_action_clip_value is not None:
                    current_actions = current_actions.clamp_(-self.final_action_clip_value, self.final_action_clip_value)
                
                denoising_chain.append(current_actions.clone())
        else:
            raise ValueError(f"Unsupported sampling method: {self.config.sampling_method}")

        final_actions = self.denorm_odo(current_actions)
        chain_tensor = torch.stack(denoising_chain, dim=1)
        
        return chain_tensor.detach(), final_actions.detach()

    def get_logprobs(
        self,
        vl_features: torch.Tensor,
        his_traj_features: torch.Tensor,
        ego_status_features: torch.Tensor,
        chains: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """Calculates the log probability of a full denoising chain."""
        B, K1, H, D = chains.shape
        num_denoising_steps = K1 - 1
        
        vl_features = self.feature_encoder(vl_features)

        his_traj_features = self.his_traj_encoder(
            his_traj_features.unsqueeze(1)          
        ).repeat(1, self.config.action_horizon, 1) 

        ego_status_features = self.ego_status_encoder(
            ego_status_features      
        )

        conditioning_embeds = {
            'vl_features': vl_features,
            'his_traj_features': his_traj_features,
            'ego_status_features': ego_status_features
        }
        
        batched_conditioning = {}
        for key, value in conditioning_embeds.items():
            batched_conditioning[key] = value.unsqueeze(1).repeat(
                1, num_denoising_steps, *(1,) * (value.ndim - 1)
            ).flatten(0, 1)

        if self.config.sampling_method == 'ddim':
            t_single = self.ddim_t[-num_denoising_steps:]
            indices_single = torch.arange(
                start=self.ddim_steps - num_denoising_steps,
                end=self.ddim_steps,
                device=chains.device
            )
            indices_batch = indices_single.repeat(B)
        else:
            t_single = torch.arange(start=K1 - 2, end=-1, step=-1, device=chains.device)
            indices_batch = None 
            
        t_batch = t_single.repeat(B)

        x_t = chains[:, :-1].reshape(-1, H, D)
        x_t_minus_1 = chains[:, 1:].reshape(-1, H, D)

        mean, logvar, _ = self.p_mean_variance(
            x_t, t_batch, indices_batch,
            batched_conditioning['vl_features'],
            batched_conditioning['his_traj_features'],
            batched_conditioning['ego_status_features'],
            deterministic=deterministic
        )

        std = torch.exp(0.5 * logvar).clamp(min=self.min_logprob_denoising_std)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(x_t_minus_1)
        
        return log_prob

    @staticmethod
    def _decode_single_path(path_tensor: torch.Tensor) -> str:
        """Decodes a 1-D long tensor of char ordinals back to a file-path string."""
        chars = []
        for code in path_tensor.cpu():
            c = code.item()
            if c == 0:
                break
            chars.append(chr(c))
        return "".join(chars)

    @staticmethod
    def _format_traj_for_lora(traj_list: list) -> str:
        """Formats a trajectory list to match Stage-2 LoRA training format exactly.
        Mirrors fmt_traj() used during LoRA fine-tuning:
            round to 2 dp → str() → strip spaces
        str(round(v,2)) drops trailing zeros (e.g. 4.3 not 4.30), which :.2f does not.
        """
        points = [[round(v, 2) for v in pt] for pt in traj_list]
        inner  = ", ".join(str(pt).replace(" ", "") for pt in points)
        return f"[{inner}]"

    def forward_grpo(
        self,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
        tokens_list,
        targets,
        sample_time: int = 8,
        deterministic=False,
        bc_coeff: float = 0.1,
        use_bc_loss: bool = True,
        image_path_tensor: Optional[torch.Tensor] = None,
        high_command_one_hot: Optional[torch.Tensor] = None,
        teacher_model=None,
        feedback_backbone=None,
    ) -> BatchFeature:
        """ELF-VLA Failure-Guided Trajectory Refinement GRPO loss."""
        self.set_frozen_modules_to_eval_mode()
        B = vl_features.shape[0]
        G = sample_time
        M = self.num_refined_samples
        self._dbg_step += 1  # debug: count forward_grpo calls

        vl_features_rep = vl_features.repeat_interleave(G, 0)
        his_traj_rep = action_input.his_traj.repeat_interleave(G, 0)
        status_feature_rep = action_input.status_feature.repeat_interleave(G, 0)

        chains, trajs = self.sample_chain(
            vl_features_rep, his_traj_rep, status_feature_rep, deterministic=False
        )

        tokens_rep = [tok for tok in tokens_list for _ in range(G)]
        unique_tokens = set(tokens_list)
        metric_cache = {}
        for token in unique_tokens:
            path = self.metric_cache_loader.metric_cache_paths[token]
            with lzma.open(path, 'rb') as f:
                metric_cache[token] = pickle.load(f)
        # rewards = self.reward_fn(trajs, tokens_rep, metric_cache)
        rewards, metric_infos = self.reward_fn(
            trajs,
            tokens_rep,
            metric_cache
        )

        # ── Policy Monitor: update EMA over ALL B×G sampled trajectories ─
        for _r in rewards.detach().cpu().tolist():
            if self._mon_ema_reward is None:
                self._mon_ema_reward = _r
            else:
                self._mon_ema_reward = (
                    self._mon_ema_decay * self._mon_ema_reward
                    + (1.0 - self._mon_ema_decay) * _r
                )
        # ─────────────────────────────────────────────────────────────────

        # ── ELF-VLA: Pass-2 Failure-Guided Refinement ────────────────────
        rewards_mat_orig = rewards.view(B, G)
        max_r     = rewards_mat_orig.max(dim=1).values   # (B,)
        fail_mask = max_r < self.failure_threshold
        refined_b = -1
        chains_ref = rewards_ref = fb_vl_rep = fb_his_rep = fb_sta_rep = None

        can_refine = (
            fail_mask.any()
            and image_path_tensor is not None
            and teacher_model is not None
            and feedback_backbone is not None
        )

        if can_refine:
            from navsim.agents.recogdrive.teacher_interface import (
                build_teacher_prompt_from_tensors,
            )
            fail_idx  = fail_mask.nonzero(as_tuple=True)[0]
            worst_b   = fail_idx[max_r[fail_idx].argmin()].item()
            worst_g   = worst_b * G + int(rewards_mat_orig[worst_b].argmin().item())
            refined_b = worst_b

            # --- [Teacher Trigger] ----------------------------------------
            _reward_before = max_r[worst_b].item()
            print(f"\n[Teacher Trigger]")
            print(f"step={self._dbg_step}")
            print(f"token={tokens_list[worst_b]}")
            print(f"reward_before={_reward_before:.4f}")
            print(f"threshold={self.failure_threshold}")
            print(f"trigger=True")
            # --------------------------------------------------------------

            image_path     = self._decode_single_path(image_path_tensor[worst_b])
            teacher_prompt = build_teacher_prompt_from_tensors(
                worst_traj_list = trajs[worst_g].cpu().tolist(),
                gt_traj_list    = targets["trajectory"][worst_b].cpu().tolist(),
                worst_metrics   = metric_infos[worst_g]["metrics"],
                command_one_hot = high_command_one_hot[worst_b].cpu().tolist(),
                status_feature  = action_input.status_feature[worst_b].cpu().tolist(),
                his_traj_4x3    = action_input.his_traj.view(B, 4, 3)[worst_b].cpu().tolist(),
            )

            _t0 = time.time()
            teacher_feedback = teacher_model.query(image_path, teacher_prompt)
            _teacher_latency = time.time() - _t0

            # --- [Teacher Feedback] ----------------------------------------
            print(f"\n[Teacher Feedback]")
            print(f"latency={_teacher_latency:.2f}s")
            print(f"chars={len(teacher_feedback)}")
            print(f"preview=\"{teacher_feedback[:300]}\"")
            with open("/tmp/teacher_log.txt", "a") as _f:
                _f.write(f"\n{'='*60}\n")
                _f.write(f"step={self._dbg_step}  token={tokens_list[worst_b]}  latency={_teacher_latency:.2f}s\n")
                _f.write(f"--- INPUT PROMPT ---\n{teacher_prompt}\n")
                _f.write(f"--- FULL OUTPUT ---\n{teacher_feedback}\n")
            # ---------------------------------------------------------------

            fault_traj_str = self._format_traj_for_lora(trajs[worst_g].cpu().tolist())
            fb_question    = (
                f"Fault Trajectory:\n{fault_traj_str}\n\n"
                f"Feedback:\n{teacher_feedback}"
            )
            pv        = load_image(image_path, max_num=12)
            n_patches = [pv.shape[0]]
            fb_device = next(feedback_backbone.parameters()).device
            pv        = pv.to(dtype=torch.bfloat16, device=fb_device)
            # Disable AMP context: feedback backbone is bf16, training AMP is fp16
            # → fp16 overflow causes NaN in hidden states without this guard.
            with torch.amp.autocast('cuda', enabled=False):
                with torch.no_grad():
                    fb_out = feedback_backbone(
                        pv.float(), [fb_question], num_patches_list=n_patches
                    )
            feedback_vl = fb_out.hidden_states[-1].to(vl_features.dtype)  # (1, seq_fb, 3584)
            # Align seq_len to match base vl_features (feedback prompt may be longer than 2800)
            target_len = vl_features.shape[1]
            fb_len     = feedback_vl.shape[1]
            if fb_len > target_len:
                feedback_vl = feedback_vl[:, :target_len, :]        # truncate
            elif fb_len < target_len:
                pad = torch.zeros(1, target_len - fb_len, feedback_vl.shape[2],
                                  dtype=feedback_vl.dtype, device=feedback_vl.device)
                feedback_vl = torch.cat([feedback_vl, pad], dim=1)  # pad

            # --- [Feedback-LoRA] -------------------------------------------
            print(f"\n[Feedback-LoRA]")
            print(f"prompt_chars={len(fb_question)}")
            print(f"hidden_shape={tuple(feedback_vl.shape)}")
            print(f"dtype={feedback_vl.dtype}")
            print(f"device={feedback_vl.device}")
            # ---------------------------------------------------------------

            # --- [Hidden Delta] --------------------------------------------
            _orig_vl = vl_features[worst_b]               # (seq_orig, 3584)
            _fb_vl   = feedback_vl[0]                      # (seq_fb,   3584)
            _seq_min = min(_orig_vl.shape[0], _fb_vl.shape[0])
            _delta   = (_orig_vl[:_seq_min].float() - _fb_vl[:_seq_min].float()).abs()
            print(f"\n[Hidden Delta]")
            print(f"mean_abs={_delta.mean().item():.6f}")
            print(f"max_abs={_delta.max().item():.6f}")
            # ---------------------------------------------------------------

            fb_vl_rep  = feedback_vl.repeat_interleave(M, 0)
            fb_his_rep = (action_input.his_traj[worst_b].unsqueeze(0)
                          .repeat_interleave(M, 0))
            fb_sta_rep = (action_input.status_feature[worst_b].unsqueeze(0)
                          .repeat_interleave(M, 0))
            chains_ref, trajs_ref = self.sample_chain(
                fb_vl_rep, fb_his_rep, fb_sta_rep, deterministic=False)
            rewards_ref, _ = self.reward_fn(
                trajs_ref, [tokens_list[worst_b]] * M, metric_cache)

            # --- Refinement Filter -------------------------------------------
            _reward_after    = rewards_ref.mean().item()
            _best_ref_reward = rewards_ref.max().item()
            _best_delta      = _best_ref_reward - _reward_before
            _accept_refined  = (_best_ref_reward > _reward_before)

            self._mon_ref_attempts += 1
            if _accept_refined:
                self._mon_ref_accepted += 1
                self._mon_best_delta_sum += _best_delta
            else:
                self._mon_ref_rejected += 1
                refined_b = -1   # fallback: merge block uses original-only path

            _ra = self._mon_ref_attempts
            _rc = self._mon_ref_accepted
            _rr = self._mon_ref_rejected
            print(f"\n[Refinement Filter]  attempts={_ra}  accepted={_rc}  rejected={_rr}  "
                  f"accepted_rate={_rc/_ra:.2%}"
                  + (f"  avg_best_delta={self._mon_best_delta_sum/_rc:+.4f}" if _rc > 0 else ""))
            # ---------------------------------------------------------------

            # --- update summary accumulators --------------------------------
            self._dbg_teacher_calls += 1
            self._dbg_latencies.append(_teacher_latency)
            self._dbg_rewards_before.append(_reward_before)
            self._dbg_rewards_after.append(_reward_after)

            # Teacher Monitor + sliding window
            _delta = _reward_after - _reward_before
            _win   = 1 if _delta > 0 else 0
            self._mon_teacher_wins += _win
            self._mon_reward_deltas.append(_delta)
            self._mon_win_deltas.append(_delta)
            self._mon_win_wins.append(_win)
            # ----------------------------------------------------------------

        # ── Merge original + refined into joint group ─────────────────────
        if refined_b != -1:
            all_c, all_v, all_h, all_s, all_r = [], [], [], [], []
            scene_sizes = []
            for b in range(B):
                sl = slice(b * G, (b + 1) * G)
                all_c.append(chains[sl]);           all_v.append(vl_features_rep[sl])
                all_h.append(his_traj_rep[sl]);     all_s.append(status_feature_rep[sl])
                all_r.append(rewards_mat_orig[b])
                if b == refined_b:
                    all_c.append(chains_ref);       all_v.append(fb_vl_rep)
                    all_h.append(fb_his_rep);       all_s.append(fb_sta_rep)
                    all_r.append(rewards_ref)
                    scene_sizes.append(G + M)
                else:
                    scene_sizes.append(G)
            chains_all  = torch.cat(all_c, 0);  vl_all  = torch.cat(all_v, 0)
            his_all     = torch.cat(all_h, 0);  sta_all = torch.cat(all_s, 0)
            rewards_all = torch.cat(all_r, 0)
        else:
            chains_all  = chains;              vl_all  = vl_features_rep
            his_all     = his_traj_rep;        sta_all = status_feature_rep
            rewards_all = rewards
            scene_sizes = [G] * B

        _refined_added = M if refined_b != -1 else 0
        if refined_b != -1:
            print(f"\n[GRPO Group] original={G} refined_added={_refined_added} final={G + _refined_added}")

        # ── Per-scene advantage normalization (supports variable group sizes) ──
        advantages = torch.zeros(rewards_all.shape[0], device=rewards_all.device)
        cursor = 0
        for gs in scene_sizes:
            r = rewards_all[cursor:cursor + gs]
            advantages[cursor:cursor + gs] = (r - r.mean()) / (r.std() + 1e-8)
            cursor += gs
        advantages = advantages.detach()

        adv_min = torch.quantile(advantages, self.clip_advantage_lower_quantile)
        adv_max = torch.quantile(advantages, self.clip_advantage_upper_quantile)
        advantages = advantages.clamp(min=adv_min, max=adv_max)

        num_denoising_steps = chains_all.shape[1] - 1
        discount = self.gamma_denoising ** torch.arange(
            num_denoising_steps - 1, -1, -1, device=advantages.device)

        # (total_samples, K) → reshape(-1) → (total_samples * K,)
        adv_weighted_flat = (
            advantages.unsqueeze(1) * discount.unsqueeze(0)
        ).reshape(-1)

        log_probs = self.get_logprobs(vl_all, his_all, sta_all,
                                      chains_all, deterministic=False)
        log_probs = log_probs.clamp(min=-5, max=2).mean(dim=[1, 2])

        policy_loss = -torch.mean(log_probs * adv_weighted_flat)
        total_loss  = policy_loss

        bc_loss = 0.0
        if use_bc_loss:
            with torch.no_grad():
                teacher_chains, _ = self.old_policy.sample_chain(
                    vl_features, action_input.his_traj, action_input.status_feature, deterministic=False
                )
            bc_logp = self.get_logprobs(vl_features, action_input.his_traj, action_input.status_feature, teacher_chains, deterministic=False)
            bc_logp = bc_logp.clamp(min=-5, max=2)
            K_steps = chains.shape[1] - 1
            bc_logp = bc_logp.view(-1, K_steps, chains.shape[2], chains.shape[3]).mean(dim=[1,2,3])
            bc_loss = -bc_logp.mean()
            total_loss = total_loss + bc_coeff * bc_loss
        # --- Monitoring logs every 50 steps -----------------------------------
        if self._dbg_step % 50 == 0:
            # [Policy Monitor]
            _ema = self._mon_ema_reward
            print(f"\n[Policy Monitor]  step={self._dbg_step}  "
                  f"ema_reward={_ema:.4f}" if _ema is not None else
                  f"\n[Policy Monitor]  step={self._dbg_step}  ema_reward=n/a")

            # [Teacher Monitor] + [Teacher Recent]
            if self._dbg_teacher_calls > 0:
                _tc   = self._dbg_teacher_calls
                _wr   = self._mon_teacher_wins / _tc
                _ad   = sum(self._mon_reward_deltas) / _tc
                _tr   = _tc / self._dbg_step
                print(f"\n[Teacher Monitor]")
                print(f"  refined_win_rate={_wr:.2%}")
                print(f"  avg_delta_reward={_ad:+.4f}")
                print(f"  trigger_rate={_tr:.2%}  ({_tc}/{self._dbg_step} steps)")

                if len(self._mon_win_deltas) > 0:
                    _n500 = len(self._mon_win_deltas)
                    _wr500 = sum(self._mon_win_wins) / _n500
                    _ad500 = sum(self._mon_win_deltas) / _n500
                    print(f"\n[Teacher Recent]  (last {_n500} calls)")
                    print(f"  refined_win_rate_500={_wr500:.2%}")
                    print(f"  avg_delta_reward_500={_ad500:+.4f}")
        # ----------------------------------------------------------------------

        return BatchFeature(data={
            "loss":           total_loss,
            "reward":         rewards.mean(),
            "policy_loss":    policy_loss,
            "bc_loss":        bc_loss,
            "refined_reward": (rewards_ref.mean() if rewards_ref is not None
                               else rewards.new_tensor(0.0)),
        })

    def print_training_health(self) -> None:
        """End-of-epoch Training Health Summary."""
        _tc = self._dbg_teacher_calls
        _ema = self._mon_ema_reward
        print("\n" + "=" * 52)
        print("[Training Health Summary]")
        print("=" * 52)
        print(f"ema_reward={_ema:.4f}" if _ema is not None else "ema_reward=n/a")
        print("")
        if _tc > 0:
            _wr  = self._mon_teacher_wins / _tc
            _ad  = sum(self._mon_reward_deltas) / _tc
            _tr  = _tc / max(self._dbg_step, 1)
            print(f"teacher_calls={_tc}")
            print(f"refined_win_rate={_wr:.2%}")
            print(f"avg_delta_reward={_ad:+.4f}")
            print(f"trigger_rate={_tr:.2%}  ({_tc}/{self._dbg_step} steps)")
            if len(self._mon_win_deltas) > 0:
                _n500  = len(self._mon_win_deltas)
                _wr500 = sum(self._mon_win_wins) / _n500
                _ad500 = sum(self._mon_win_deltas) / _n500
                print("")
                print(f"recent_refined_win_rate (last {_n500})={_wr500:.2%}")
                print(f"recent_avg_delta_reward (last {_n500})={_ad500:+.4f}")
        else:
            print("teacher_calls=0  (teacher not triggered this epoch)")

        _ra = self._mon_ref_attempts
        if _ra > 0:
            _rc  = self._mon_ref_accepted
            _rr  = self._mon_ref_rejected
            print("")
            print("[Refinement Filter Summary]")
            print(f"  attempts={_ra}")
            print(f"  accepted={_rc}")
            print(f"  rejected={_rr}")
            print(f"  accepted_rate={_rc/_ra:.2%}")
            if _rc > 0:
                print(f"  avg_best_delta={self._mon_best_delta_sum/_rc:+.4f}")
        print("=" * 52 + "\n")

    def norm_odo(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Normalizes trajectory coordinates and heading to the range [-1, 1]."""
        x = 2 * (trajectory[..., 0:1] + 1.57) / 66.74 - 1
        y = 2 * (trajectory[..., 1:2] + 19.68) / 42 - 1
        heading = 2 * (trajectory[..., 2:3] + 1.67) / 3.53 - 1
        return torch.cat([x, y, heading], dim=-1)
    
    def denorm_odo(self, normalized_trajectory: torch.Tensor) -> torch.Tensor:
        """Denormalizes trajectory from [-1, 1] back to original coordinate space."""
        x = (normalized_trajectory[..., 0:1] + 1) / 2 * 66.74 - 1.57
        y = (normalized_trajectory[..., 1:2] + 1) / 2 * 42 - 19.68
        heading = (normalized_trajectory[..., 2:3] + 1) / 2 * 3.53 - 1.67
        return torch.cat([x, y, heading], dim=-1)

    # def reward_fn(
    #     self,
    #     pred_traj: torch.Tensor,
    #     tokens_list,
    #     cache_dict,
    # ) -> torch.Tensor:
    #     """Calculates PDM scores for a batch of predicted trajectories."""
    #     pred_np = pred_traj.detach().cpu().numpy()
    #     rewards = []
    #     for i, token in enumerate(tokens_list):
    #         trajectory = Trajectory(pred_np[i])
    #         metric_cache = cache_dict[token]
    #         pdm_result = pdm_score(
    #             metric_cache=metric_cache,
    #             model_trajectory=trajectory,
    #             future_sampling=self.simulator.proposal_sampling,
    #             simulator=self.simulator,
    #             scorer=self.train_scorer,
    #         )
    #         rewards.append(asdict(pdm_result)["score"])
    #     return torch.tensor(rewards, device=pred_traj.device, dtype=pred_traj.dtype).detach()
    def reward_fn(
        self,
        pred_traj,
        tokens_list,
        cache_dict,
    ):
        pred_np = pred_traj.detach().cpu().numpy()

        rewards = []
        metric_infos = []

        for i, token in enumerate(tokens_list):

            trajectory = Trajectory(pred_np[i])
            metric_cache = cache_dict[token]

            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=self.simulator.proposal_sampling,
                simulator=self.simulator,
                scorer=self.train_scorer,
            )

            result_dict = asdict(pdm_result)

            rewards.append(result_dict["score"])

            metric_infos.append({
                "token": token,

                # ===== full metrics =====
                "metrics": result_dict,

                # ===== ego =====
                "ego_speed": float(
                    metric_cache.ego_state.dynamic_car_state.speed
                ) if hasattr(metric_cache.ego_state.dynamic_car_state, "speed") else None,

                "steering": float(
                    metric_cache.ego_state.tire_steering_angle
                ),

                # ===== route =====
                "route_lane_ids": metric_cache.route_lane_ids,

                # ===== centerline =====
                "centerline_len": len(
                    metric_cache.centerline._states_se2_array
                ),

                # ===== drivable =====
                "drivable_tokens": metric_cache.drivable_area_map._tokens,
            })

        rewards = torch.tensor(
            rewards,
            device=pred_traj.device,
            dtype=pred_traj.dtype
        ).detach()

        return rewards, metric_infos
    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

#https://github.com/irom-princeton/dppo/blob/cc7234ad7ff39a8f32de3af903606723a16f0648/model/diffusion/eta.py#L12
class EtaFixed(nn.Module):

    def __init__(
        self,
        base_eta=0.5,
        min_eta=0.1,
        max_eta=1.0,
        **kwargs,
    ):
        super().__init__()
        self.eta_logit = nn.Parameter(torch.ones(1))
        self.min = min_eta
        self.max = max_eta

        self.eta_logit.data = torch.atanh(
            torch.tensor([2 * (base_eta - min_eta) / (max_eta - min_eta) - 1])
        )

    def __call__(self, x):
        """Match input batch size, but do not depend on input"""
        B = len(x)
        device = x.device
        eta_normalized = torch.tanh(self.eta_logit)

        eta = 0.5 * (eta_normalized + 1) * (self.max - self.min) + self.min
        return torch.full((B, 1), eta.item()).to(device)