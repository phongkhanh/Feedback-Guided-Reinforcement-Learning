import json
import os
import socket
import struct
import sys
import torch
from transformers import AutoModel, AutoTokenizer

# create_prompt.py lives at the project root — add it to path once so it can
# be imported without running its __main__ block.
_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from create_prompt import build_teacher_prompt  # noqa: E402
from navsim.agents.recogdrive.utils.internvl_preprocess import load_image


def build_teacher_prompt_from_tensors(
    worst_traj_list: list,
    gt_traj_list: list,
    worst_metrics: dict,
    command_one_hot: list,
    status_feature: list,   # layout: [cmd(3), vel(2), acc(3)]
    his_traj_4x3: list,     # [[x,y,h] × 4 timesteps]
) -> str:
    """
    Reconstructs the teacher prompt from training tensors.
    Mirrors the item/sample/ego dict structure that create_prompt.build_teacher_prompt
    expects, filling in what is available from the batch and gracefully omitting
    scene-relation info (not cached at RL training time).
    """
    # yaw rate estimated from consecutive heading values (0.5 s interval)
    yaw_rate = float((his_traj_4x3[-1][2] - his_traj_4x3[-2][2]) / 0.5)
    ego = [
        float(status_feature[3]),   # speed  (vx, forward component)
        float(status_feature[5]),   # acc_x
        float(status_feature[6]),   # acc_y
        yaw_rate,
    ]
    item = {
        "traj_GT":       gt_traj_list,
        "traj_bad":      worst_traj_list,
        "worst_metrics": worst_metrics,
    }
    sample = {
        "driving_command": command_one_hot,   # list of 3 floats, np.argmax works
        "anns": {},                           # scene relations unavailable at RL time
    }
    return build_teacher_prompt(item, sample, ego)


class TeacherModelClient:
    """
    Lightweight client for multi-GPU RL training.
    Connects to teacher_server.py running on a dedicated GPU via TCP socket.
    Each training rank creates its own client connection.
    """

    def __init__(self, host: str = "localhost", port: int = 55123):
        self.host = host
        self.port = port
        print(f"[TeacherModelClient] Will connect to {host}:{port} on demand.")

    @staticmethod
    def _send(conn, obj):
        data = json.dumps(obj).encode()
        conn.sendall(struct.pack(">I", len(data)) + data)

    @staticmethod
    def _recv(conn):
        raw = b""
        while len(raw) < 4:
            chunk = conn.recv(4 - len(raw))
            if not chunk:
                raise ConnectionError("Teacher server closed connection")
            raw += chunk
        length = struct.unpack(">I", raw)[0]
        data = b""
        while len(data) < length:
            data += conn.recv(length - len(data))
        return json.loads(data.decode())

    def query(self, image_path: str, prompt_text: str) -> str:
        with socket.create_connection((self.host, self.port), timeout=300) as conn:
            self._send(conn, {"image_path": image_path, "prompt_text": prompt_text})
            resp = self._recv(conn)
        return resp.get("feedback", "")


class TeacherModel:
    """
    Wraps InternVL3-38B exactly as test_VLM.py does, but as a persistent
    callable object pinned to a dedicated GPU (default cuda:1 = physical GPU0).
    """

    def __init__(self, model_path: str, device: str = "cuda:1"):
        print(f"[TeacherModel] Loading from {model_path} on {device} ...")
        # Mirror test_VLM.py exactly: device_map="auto" lets InternVL3 handle
        # its internal mixed precision (ViT=fp16 / LLM=bf16) correctly.
        # max_memory restricts the model to only the teacher GPU.
        device_idx = int(device.split(":")[-1])
        n_gpu = torch.cuda.device_count()
        max_memory = {
            i: (0 if i != device_idx else "79GiB")
            for i in range(n_gpu)
        }
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto",          # same as test_VLM.py
            max_memory=max_memory,      # restrict to teacher GPU only
        ).eval()
        # test_VLM.py runs with CUDA_VISIBLE_DEVICES=1 (1 GPU) → ViT+LLM
        # both land on same device in bf16. With 2 GPUs visible, device_map="auto"
        # may leave ViT in fp16 while LLM is bf16 → mismatch at generate() line 334.
        # Fix: force ViT to bf16 to match LLM.
        if hasattr(self.model, 'vision_model'):
            self.model.vision_model.to(torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        self._gen_cfg = dict(max_new_tokens=256, do_sample=False)
        self._device = device
        print("[TeacherModel] Ready.")

    @torch.no_grad()
    def query(self, image_path: str, prompt_text: str) -> str:
        """Returns teacher feedback string. Mirrors test_VLM.py lines 161-165."""
        # pixel_values → bfloat16 + teacher GPU, same as test_VLM.py
        pixel_values = (
            load_image(image_path, max_num=12)
            .to(torch.bfloat16)
            .to(self._device)
        )
        return self.model.chat(
            self.tokenizer,
            pixel_values,
            "<image>\n" + prompt_text,
            self._gen_cfg,
        )
