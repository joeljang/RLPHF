import json
from datasets import load_dataset, ReadInstruction
from scipy import stats
from transformers import LlamaTokenizer, AutoModelForCausalLM, set_seed, GenerationConfig
from accelerate import Accelerator
import random
import argparse
import torch
import functools
import multiprocessing
from tqdm import tqdm
import os
from trl import AutoModelForCausalLMWithValueHead
from peft import get_peft_model, TaskType, LoraConfig
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
LOCAL_WORLD_SIZE = local_rank = int(os.environ["LOCAL_WORLD_SIZE"])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="./data/alpaca_gpt4_data.json")
    parser.add_argument("--start_per", type=int, default=0)
    parser.add_argument("--end_per", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="generations/test.json")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--decoding_temperature", type=int, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_return_sequences", type=int, default=2)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument('--checkpoint_dirs', action='append', default=None)
    parser.add_argument("--gpu_num", type=int, default=0)

    return parser.parse_args()

def generate_prompt(personalized_instruction: str, instruction: str, input_ctxt: str = None) -> str:
    if input_ctxt!="":
        return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n### Instruction:\n{instruction} {personalized_instruction}\n### Input:\n{input_ctxt}\n### Response:"
    else:
        return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n### Instruction:\n{instruction} {personalized_instruction}\n### Response:"

def generate_prompt_tulu(personalized_instruction: str, instruction: str, input_ctxt: str = None) -> str:
    return f"<|user|>\n{instruction} {input_ctxt} {personalized_instruction}\n<|assistant|>\n"

def main(args):
    if args.checkpoint_dirs!=None:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.base_model,
            load_in_8bit=True,
            peft_config=lora_config,
        )
        ckpt_params_lst = []
        for checkpoint_dir in args.checkpoint_dirs:
            ckpt_params = torch.load(checkpoint_dir, map_location=f'cuda:{args.gpu_num}')
            ckpt_params_lst.append(ckpt_params)
        loading_cnt = 0
        for name, param in model.named_parameters():
            temp_name = name.replace('pretrained_model.', '')
            temp_name = temp_name.replace('.default', '')
            if temp_name in ckpt_params:
                param_sum = None
                for i in range(len(ckpt_params_lst)):
                    params = ckpt_params_lst[i][temp_name]
                    if param_sum == None:
                        param_sum = params
                    else:
                        param_sum += params
                final_param = param_sum / len(ckpt_params_lst)
                param.data = final_param
                loading_cnt+=1
        print(f'loaded {loading_cnt} LoRA weights!')
        model.to('cuda')
    elif args.checkpoint_dir!=None:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.base_model,
            load_in_8bit=True,
            peft_config=lora_config,
        )
        ckpt_params = torch.load(args.checkpoint_dir, map_location='cuda:0')
        loading_cnt = 0
        for name, param in model.named_parameters():
            temp_name = name.replace('pretrained_model.', '')
            temp_name = temp_name.replace('.default', '')
            if temp_name in ckpt_params:
                param.data = ckpt_params[temp_name]
                loading_cnt+=1
        print(f'loaded {loading_cnt} LoRA weights!')
        model.to('cuda')
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            load_in_4bit=True,
            device_map='auto'
        )
    model.eval()    
    tokenizer = LlamaTokenizer.from_pretrained(args.base_model, use_fast=False)
    tokenizer.padding_side = 'left'

    #Loading the configs to use for generation
    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        temperature=args.decoding_temperature,
        do_sample=True,
        early_stopping=True,
        num_return_sequences=args.num_return_sequences
    )
    # Load data to perform inference on 
    data_path = args.dataset_name
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=data_path, split = ReadInstruction(args.split, from_=args.start_per, to=args.end_per, unit='%', rounding='pct1_dropremainder'))
    else:
        dataset = load_dataset(data_path, split=args.split)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size)

    entries = []
    indx = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            input_entries = []
            instructions = []
            inputs = []
            for j in range(len(batch['input'])):
                if j % LOCAL_WORLD_SIZE == LOCAL_RANK:
                    instructions.append(batch['instruction'][j])
                    inputs.append(batch['input'][j])
                    if 'tulu' in args.base_model:
                        input_entry = generate_prompt_tulu(args.prompt, batch['instruction'][j], batch['input'][j])
                    else:
                        input_entry = generate_prompt(args.prompt, batch['instruction'][j], batch['input'][j])
                    input_entries.append(input_entry)
            input_ids = tokenizer(input_entries, max_length=args.max_seq_length, padding=True, truncation=True, return_tensors="pt").input_ids
            input_ids = input_ids.to('cuda')
            outputs = model.generate(
                input_ids = input_ids,
                generation_config = generation_config,
                max_new_tokens = args.max_seq_length,
                return_dict_in_generate = True
            )
            outputs_text = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            
            for j in range(len(input_entries)):
                input_text = input_entries[j]
                o1_indx = j * 2
                o2_indx = (j * 2) + 1
                output_1 = outputs_text[o1_indx][len(input_text):]
                output_2 = outputs_text[o2_indx][len(input_text):]
                '''
                if indx % 100 == 0:
                    print(f'INPUT: {input_text}')
                    print(f'OUTPUT #1: ', output_1)
                    print(f'OUTPUT #2: ', output_2)
                '''
                print(f'INPUT: {input_text}')
                print(f'OUTPUT #1: ', output_1)
                print(f'OUTPUT #2: ', output_2)
                print(indx)
                entry = {
                    "id": indx,
                    "instruction": instructions[j],
                    "input": inputs[j],
                    "preference_prompt": args.prompt,
                    "output_1": output_1,
                    "output_2": output_2
                }
                entries.append(entry)
                indx+=1

    #with open(args.output_dir + f'_{LOCAL_RANK}.json', "w") as f:
    #    json.dump(entries, f)
    with open(args.output_dir, "w") as f:
        json.dump(entries, f)

if __name__ == "__main__":
    args = get_args()
    assert args.base_model != "", "Please provide the llama model path"
    set_seed(args.seed)
    main(args)