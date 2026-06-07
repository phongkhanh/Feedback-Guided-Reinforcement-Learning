eval 
HYDRA_FULL_ERROR=1 sh cache_dataset/run_metric_caching.sh

HYDRA_FULL_ERROR=1 sh evaluation/run_recogdrive_agent_pdm_score_evaluation_2b.sh

Diffusion Planner Imitation Learning

HYDRA_FULL_ERROR=1 sh cache_dataset/run_caching_recogdrive_hidden_state.sh
HYDRA_FULL_ERROR=1 sh training/run_recogdrive_train_multi_node_2b.sh

train RL
HYDRA_FULL_ERROR=1 sh cache_dataset/run_metric_caching_train.sh
HYDRA_FULL_ERROR=1 sh training/run_recogdrive_train_multi_node_rl_2b.sh


---------------------
8b
HYDRA_FULL_ERROR=1 sh cache_dataset/run_metric_caching.sh

HYDRA_FULL_ERROR=1 sh evaluation/run_recogdrive_agent_pdm_score_evaluation_8b.sh

Diffusion Planner Imitation Learning
HYDRA_FULL_ERROR=1 sh cache_dataset/run_caching_recogdrive_hidden_state_8b.sh
HYDRA_FULL_ERROR=1 sh training/run_recogdrive_train_multi_node_8b.sh

train RL
HYDRA_FULL_ERROR=1 sh cache_dataset/run_metric_caching_train.sh
HYDRA_FULL_ERROR=1 sh training/run_recogdrive_train_multi_node_rl_8b.sh