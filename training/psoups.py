import os

import torch
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import (
    Adafactor,
    AutoTokenizer,
    LlamaTokenizer,
    HfArgumentParser,
    pipeline,
    AutoModelForSequenceClassification
)
import torch.nn as nn

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
import wandb
from scipy import stats as scipy_stats
import math

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    rm_model_name: Optional[str] = field(default=None, metadata={"help": "the rm model name (in case different from the policy model)"})
    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    dataset_name: Optional[str] = field(default="", metadata={"help": "the dataset name"})
    val_dataset_name: Optional[str] = field(default=None, metadata={"help": "the dataset name"})
    val_every_n_steps: Optional[int] = field(default=10, metadata={"help": "how many graident update steps to do before getting validation RM scores"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    max_length: Optional[int] = field(default=512, metadata={"help": "maximum length for input"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="./checkpoints/tuning_llama_rl/",
                                      metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    wandb_project: Optional[str] = field(
    default='default_project',
    metadata={
        "help": "wandb project name"
    },
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "wandb run name"
        },
    )


parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
run = wandb.init(project=script_args.wandb_project, group = script_args.wandb_run_name)

set_seed(script_args.seed)

def freeze_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            print(name)
            trainable_params += param.numel()
            param.requires_grad = False
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}. Now froze all params."
    )

def get_reward_stats(lst):
    lst = torch.stack(lst)
    v_min, v_max, v_mean = lst.min(), lst.max(), torch.mean(lst)
    return v_min, v_max, v_mean

PREF_PROMPTS = [
    "Generate a response that can be easily understood by an elementary school student.",
    "Generate a response that only a PhD Student in that specific field could understand.",
    "Generate a response that is concise and to the point, without being verbose.",
    "Generate a response that is very informative, without missing any background information.",
    "Generate a response that is friendly, witty, funny, and humorous, like a close friend.",
    "Generate a response in an unfriendly manner.",
    "Generate a response in a sassy manner.",
    "Generate a response in a sarcastic manner."
]
# Designating preference combination
if '1A' in script_args.wandb_run_name:
    PREF = 0
elif '1B' in script_args.wandb_run_name:
    PREF = 1
elif '2A' in script_args.wandb_run_name:
    PREF = 2
elif '2B' in script_args.wandb_run_name:
    PREF = 3
elif '3A' in script_args.wandb_run_name:
    PREF = 4
elif '3B' in script_args.wandb_run_name:
    PREF = 5
elif '3C' in script_args.wandb_run_name:
    PREF = 6
elif '3D' in script_args.wandb_run_name:
    PREF = 7
else:
    raise Exception('the wandb run name does not contain any indicator for RM being used.')

def add_preference_prompt_during_val(query):
    query = query.replace('\n<|assistant|>\n', '')
    query = query + f" {PREF_PROMPTS[PREF]}\n<|assistant|>\n"
    return query

# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(
        tokenizer, dataset_name, input_min_text_length=2, input_max_text_length=8
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    data_path = dataset_name
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        train_dataset = load_dataset("json", data_files=data_path, split='train')
    else:
        train_dataset = load_dataset(data_path, split='train')
    original_columns = train_dataset.column_names
    num_proc = 24

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for instruction, input_ctxt in zip(examples["instruction"], examples["input"]):
            query = f"<|user|>\n{instruction} {input_ctxt} \n<|assistant|>\n"
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) < script_args.max_length, batched=False)

    ds.set_format(type="torch")
    return ds

def build_val_dataset(
        tokenizer, dataset_name, input_min_text_length=2, input_max_text_length=8
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    data_path = dataset_name
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        train_dataset = load_dataset("json", data_files=data_path, split='train')
    else:
        train_dataset = load_dataset(data_path, split='train')
    original_columns = train_dataset.column_names
    num_proc = 24

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for instruction, input_ctxt in zip(examples["instruction"], examples["input"]):
            query = f"<|user|>\n{instruction} {input_ctxt} {PREF_PROMPTS[PREF]}\n<|assistant|>\n"
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) < script_args.max_length, batched=False)

    ds.set_format(type="torch")
    return ds

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


reward_model_name = script_args.reward_model_name
config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
    use_score_scaling = True,
    use_score_norm = True
)

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
rw_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 16,
    "truncation": True
}

if "decapoda" in script_args.model_name.lower():
    tokenizer = LlamaTokenizer.from_pretrained(script_args.model_name, use_fast = False)
    # required for llama
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        }
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, use_fast = False)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

# Now let's build the model, the reference model, and the tokenizer.
current_device = Accelerator().local_process_index
LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"])

# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(tokenizer, script_args.dataset_name)
#shuffle dataset
dataset = dataset.shuffle(seed=script_args.seed)
if script_args.val_dataset_name:
    val_dataset = build_val_dataset(tokenizer, script_args.val_dataset_name)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = script_args.batch_size * LOCAL_WORLD_SIZE, collate_fn=collator)



lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    load_in_4bit=True,
    device_map={"": current_device},
    peft_config=lora_config,
)

optimizer = None
if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug

if script_args.rm_model_name == None:
    script_args.rm_model_name = script_args.model_name


reward_model = AutoModelForSequenceClassification.from_pretrained(
    script_args.rm_model_name, num_labels=1, torch_dtype=torch.bfloat16,
    load_in_4bit=True
)
rm_peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=True,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

reward_model = get_peft_model(reward_model, rm_peft_config)
ckpt_params = torch.load(reward_model_name, map_location=f'cuda:{current_device}')
loading_cnt = 0
for name, param in reward_model.named_parameters():
    temp_name = name.replace('.default', '')
    temp_name = temp_name.replace('.original_module', '')
    if temp_name in ckpt_params:
        param.data = ckpt_params[temp_name]
        loading_cnt+=1

print(f'loaded {loading_cnt} LoRA weights for the reward model.')
reward_model.to(f'cuda:{current_device}')
freeze_trainable_parameters(reward_model)

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    # "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": 100_000,
}
output_min_length = 32
output_max_length = script_args.output_max_length
output_length_sampler = LengthSampler(output_min_length, output_max_length)

for steps, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    question_tensors = batch["input_ids"]

    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,
        length_sampler=output_length_sampler,
        **generation_kwargs,
    )
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    input_ids = tokenizer(texts, max_length=script_args.max_length ,return_tensors="pt", padding=True, truncation=True).input_ids
    reward_outputs = reward_model(input_ids = input_ids.to(reward_model.device))[0]
    #reward_outputs = nn.functional.logsigmoid(reward_outputs)
    rewards = [torch.tensor(output[0].float()) - script_args.reward_baseline for output in reward_outputs]
    v_min, v_max, v_mean = get_reward_stats(rewards)
    # Run PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
    #ppo_trainer.log_stats(stats, batch, rewards)
    # Getting RM mean on validation instances
    print(f'Finished {steps} steps of PPO outer loop')
    # First log stats during training
    wandb.log({"train_reward_mean": v_mean})
    wandb.log({"kl": stats["objective/kl"]})
    wandb.log({"kl_coef": stats["objective/kl_coef"]})
    wandb.log({"steps": steps})
    #if False:
    if script_args.val_dataset_name:
        if steps % script_args.val_every_n_steps == 0:
            text_table = wandb.Table(columns=["step", "query", "response", "reward"])
            print(f'Logging train & val stats in this time step..')
            with torch.no_grad():
                rewards_lst = None
                for i, val_batch in tqdm(enumerate(val_dataloader)):
                    print(i)
                    question_tensors = val_batch["input_ids"]
                    question_tensors = [question_tensors[j] for j in range(len(question_tensors)) if j % LOCAL_WORLD_SIZE == current_device]
                    if len(question_tensors) != 0:
                        response_tensors = ppo_trainer.generate(
                            question_tensors,
                            return_prompt=False,
                            length_sampler=output_length_sampler,
                            **generation_kwargs,
                        )
                        val_batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
                        val_batch["original_query"] = tokenizer.batch_decode(question_tensors, skip_special_tokens=True)
                        # Compute sentiment score
                        texts = []
                        for q, r in zip(val_batch["original_query"], val_batch["response"]):
                            text = q + r
                            texts.append(text)
                        input_ids = tokenizer(texts, max_length=script_args.max_length ,return_tensors="pt", padding=True, truncation=True).input_ids
                        reward_outputs = reward_model(input_ids = input_ids.to(reward_model.device))[0]
                        rewards = [torch.tensor(output[0]).cpu().float() - script_args.reward_baseline for output in reward_outputs]
                        for j in range(len(rewards)):
                            text_table.add_data(steps, val_batch["original_query"][j], val_batch["response"][j], rewards[j])
                        if rewards_lst == None:
                            rewards_lst = rewards 
                        else:
                            rewards_lst = rewards_lst + rewards
            print(f'finished validating.. logging stats!')
            val_reward_stats = scipy_stats.describe(rewards_lst)
            # Logging stats during validation
            wandb.log({"val_reward_mean": val_reward_stats.mean})
            wandb.log({"val_reward_std": math.sqrt(val_reward_stats.variance)})
            run.log({"val_instance_samples": text_table})
        else:
            print(f'Not logging anything in this time step..')
    
                    

    if script_args.save_freq and steps and steps % script_args.save_freq == 0:
        save_path = script_args.output_dir + f"step_{steps}"
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)