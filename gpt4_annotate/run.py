# Currently using personal api key for debugging
import os
import argparse

from alpaca_farm.utils import jload
from alpaca_farm.auto_annotations import PairwiseAutoAnnotator
import tiktoken

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--open_ai_key", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default="examples/data/outputs_pairs.json")
    parser.add_argument("--annotators", type=str, default="annotators/test/configs.yaml")
    parser.add_argument("--saving_path", type=str, default="examples/data/annotations.json")
    return parser.parse_args()

def calculate_rate(args, data):
    # Set-up the tokenizer 
    tokenizer = tiktoken.get_encoding("cl100k_base")
    total_token_num = 0
    for row in data:
        text_input = str(row['instruction']) + str(row['input']) + str(row['output_1']) + str(row['output_2'])
        tokens = tokenizer.encode(text_input)
        total_token_num += len(tokens)
    if 'gpt-3.5-turbo' in args.annotators or 'chatgpt' in args.annotators:
        price_per_1k = 0.0015
    elif 'text-davinci-003' in args.annotators or 'gpt4' in args.annotators:
        price_per_1k = 0.03
    elif 'annotator_pool_v0' in args.annotators:
        price_per_1k = 0.03 # Estimate for Alpacafarm annotator pool w/ 14 annotator combinations
    else:
        raise Exception('Please include the model name in the annotators config.')
    estimated_pricing = (total_token_num / 1000 ) * price_per_1k
    estimated_pricing_with_demonstrations = estimated_pricing * 4 # Estimated price for the prompts appeneded
    print(f'The esimtated cost for this annotation is ${estimated_pricing_with_demonstrations} with {total_token_num} number of tokens (excluding demonstrations).')
    #consent = input(f'The esimtated cost for this annotation is ${estimated_pricing_with_demonstrations} with {total_token_num} number of tokens (excluding demonstrations). Would you like to go ahead? \nReply with y/n: ')
    #if consent == 'y':
    #    print('proceeding with the annotation!')
    #else:
    #    exit('not proceeding with the annotation because it is too expensive..! :(')

def main(args):
    if args.open_ai_key:
        decoding_kwargs = dict(
            openai_api_key = args.open_ai_key,
            openai_organization_ids = None, # ["org-...","org-..."] you can set multiple orgs to avoid rate limits
        )
    else:
        raise Exception('Please provide the OpenAI API Key!')

    annotator = PairwiseAutoAnnotator(annotators_config = args.annotators, saving_path = args.saving_path, **decoding_kwargs)

    samples = jload(args.input_dir)
    calculate_rate(args, samples)
    annotated = annotator.annotate_pairs(samples)
    pref1_cnt=0
    pref2_cnt=0
    same = 0
    for a in annotated:
        if a['preference'] == 1:
            pref1_cnt += 1
        elif a['preference'] == 2:
            pref2_cnt += 1
        else:
            same += 1
    print(f'The total num of pref 1 {pref1_cnt}')
    print(f'The total num of pref 2 {pref2_cnt}')
    print(f'The total num of same preference {same}')

if __name__ == "__main__":
    args = get_args()
    main(args)