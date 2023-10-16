<h1 align="center">RLPHF</h1>

This is the official github repository for Personalized Soups: Personalized Large Language Model Alignment via Post-hoc Parameter Merging.

### Setup

Install dependencies

```
pip install -r requirements.txt
```

### How to use?

- **Step 1 - Generate Rollouts**

```
torchrun --nnodes 1 --nproc_per_node 1 /net/nfs.cirrascale/mosaic/joel/personalized-rlhf/generate_rollouts.py \
    --output_dir $OUTPUT_DIR \
    --base_model $PATH_TO_TULU_CKPT \
    --dataset_name 'data/alpaca_gpt4_10k.json' \
    --prompt 'Generate a response that can be easily understood by an elementary school student.' \
    --batch_size 16 \
    --start_per 0 \
    --end_per 100
```

To get the Tulu checkpoints, refer to [this repository](https://arxiv.org/abs/2302.03202). Feel free to put any customized prompt from the prompt config.

- **Step 2 - Label generated rollouts using GPT4**
```
cd gpt4_annotate;
python run.py --open_ai_key $OPEN_AI_KEY \
	--input_dir $ROLLOUT_DIR \
	--saving_path $SAVE_PATH \
	--annotators $YAML_FILE_OF_ANNOTATOR_CONFIG
```
the .yaml file of the GPT4 annotator configs used for our experiments are provided in the GPT4_b5 directory. First clone [https://github.com/tatsu-lab/alpaca_farm.git](https://github.com/tatsu-lab/alpaca_farm.git). Next place the GPT4_b5 directory inside alpaca_farm/auto_annotations/annotators and refer to the target .yaml file (e.g. pref1A.yaml) with the --annotators config. Please refer to the Alpacafarm code repo for more details. 

- **Step 3 - Training Reward Model**
Next, we utilize the GPT4 annotation for reward model training. 
An example script is provided below:
```
torchrun --nnodes 1 --nproc_per_node 4 training_reward_model.py 
    --model_name $PATH_TO_TULU_CKPT \
    --dataset_name $PATH_TO_RM_DATA \
    --eval_dataset_name $EVAL_DATASET_NAME \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 2 \
    --num_train_epochs 1 \
    --wandb_project $WANDB_PROJECT_NAME \
    --wandb_run_name $WANDB_RUN_NAME
```

You can find the list of reward model training data in the data/rm_training directory. You can choose to create your own, custom eval dataset during rm training.

- **Step 4 - Policy Model Training**
Here are sample script rns you can use to train each models:
Traditional RLHF
```
torchrun --nnodes 1 --nproc_per_node 4 training/rlhf.py \
    --dataset_name 'data/alpaca_gpt4_10k.json' \
    --model_name $PATH_TO_TULU_CKPT \
    --reward_model_name $DIR_TO_RM \
    --output_dir $OUTPUT_DIR \
    --adafactor False --save_freq 10 --output_max_length 512 --batch_size 16 --gradient_accumulation_steps 8 --batched_gen True --ppo_epochs 8 --learning_rate 1.4e-5 --mini_batch_size 2 \
    --early_stopping True --log_with wandb --val_dataset_name 'data/koala_eval_50_.json' --val_every_n_steps 10 \
    --wandb_project $WANDB_PROJECT_NAME --wandb_run_name $WANDB_RUN_NAME  \
```

$DIR_TO_RM is the directory to the adapter_model.bin from the reward model training output directory.

Multitask Training
```
torchrun --nnodes 1 --nproc_per_node 4 training/multitask_training.py \
    --base_model $PATH_TO_TULU_CKPT \
    --dataset_name 'data/alpca_gpt4_10k_mt.json' \
    --streaming --lr_scheduler_type 'constant' \
    --learning_rate 1e-5 --max_steps 1000 \
    --output_dir $OUTPUT_DIR \
    --project_name $WANDB_PROJECT_NAME \
    --run_name $WANDB_RUN_NAME
```

P-MORL
```
torchrun --nnodes 1 --nproc_per_node 8 /net/nfs.cirrascale/mosaic/joel/personalized-rlhf/ppo_training/tuning_lm_with_rl_policy_4_ablation.py --dataset_name '/net/nfs.cirrascale/mosaic/joel/personalized-rlhf/data/train/alpaca_gpt4_data_512_augmented_multiple_10k_ablation.json' --log_with wandb --model_name '/net/nfs.cirrascale/mosaic/joel/tulu-7b-recovered' --reward_model_name '/net/nfs.cirrascale/mosaic/joel/personalized-rlhf/checkpoints/final_ckpts/RM_all_pref_ablation_10k_prompted/peft_last_checkpoint/adapter_model.bin' --adafactor False --save_freq 10 --output_max_length 512 --batch_size 16 --gradient_accumulation_steps 4 --batched_gen True --ppo_epochs 8 --learning_rate 1.4e-5 --early_stopping True --output_dir '/net/nfs.cirrascale/mosaic/joel/personalized-rlhf/checkpoints/final_ckpts_update/policy_ablation_4_new/' --wandb_project policy_training_7b_update --wandb_run_name policy_4_ablation --mini_batch_size 2 --val_dataset_name '/net/nfs.cirrascale/mosaic/joel/personalized-rlhf/data/eval/koala_50_during_train.json' --val_every_n_steps 10
```

P-Soups
```
torchrun --nnodes 1 --nproc_per_node 4 /net/nfs.cirrascale/mosaic/joel/personalized-rlhf/ppo_training/tuning_lm_with_rl_policy_7.py --dataset_name '/net/nfs.cirrascale/mosaic/joel/personalized-rlhf/data/train/alpaca_gpt4_data_512_10k.json' --log_with wandb --model_name '/net/nfs.cirrascale/mosaic/joel/tulu-7b-recovered' --reward_model_name '/net/nfs.cirrascale/mosaic/joel/personalized-rlhf/checkpoints/final_ckpts/RM_pref1A_10k_prompted/RM_pref1A_prompted/checkpoint-5434/adapter_model.bin' --adafactor False --save_freq 10 --output_max_length 512 --batch_size 16 --gradient_accumulation_steps 8 --batched_gen True --ppo_epochs 8 --learning_rate 1.4e-5 --early_stopping True --output_dir '/net/nfs.cirrascale/mosaic/joel/personalized-rlhf/final_ckpts_prompted/policy_7_noprompts_pref_1A/' --wandb_project policy_training_7b_prompted --wandb_run_name policy_7_noprompts_pref1A --mini_batch_size 2 --val_dataset_name '/net/nfs.cirrascale/mosaic/joel/personalized-rlhf/data/eval/koala_50_during_train.json' --val_every_n_steps 10
```
