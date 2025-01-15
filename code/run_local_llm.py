import os
import re
import json
import random
import datetime
import base64
import time
import requests
import traceback
import argparse
from collections import defaultdict

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    PeftModel,
    PeftConfig,
    get_peft_model
)
from datasets import Dataset, load_dataset

from pathlib import Path
from tqdm import tqdm

from src.utils import load_json, save_json, mental_verb
from src.nn import load_model_hf
from prompts import cot_prompts


idx_to_option = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
}

data_to_num_options = {
    'tomato': 4,
    'siqa': 3
}

sotopia_agent = {a['pk']: a for a in load_json('data/sotopia/agent.json')}
sotopia_combo = {a['pk'][-5:]: a for a in load_json('data/sotopia/combo.json')}
sotopia_environment = {a['pk']: a for a in load_json('data/sotopia/environment.json')}


def ArgParser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--method", default="", type=str)
    parser.add_argument("--data_type", default="siq", type=str, help="")
    parser.add_argument("--test_files", default=[], nargs='*', type=str, help="")
    parser.add_argument("--exclude_options_from_prompt", default=False, action="store_true", help="")
    parser.add_argument("--cot_prompt", default=None, type=str)
    parser.add_argument("--output_dir", default="output", type=str, help="")
    parser.add_argument("--output_suffix", default="", type=str, help="")
    parser.add_argument("--do_gen_qa", default=False, action="store_true", help="")
    parser.add_argument("--do_mc_qa", default=False, action="store_true", help="")
    parser.add_argument("--do_sample", default=False, action="store_true", help="")
    parser.add_argument("--top_p", default=0.9, type=float, help="")
    parser.add_argument("--temperature", default=0.6, type=float, help="")
    parser.add_argument("--exp_id", default="default", type=str)
    parser.add_argument("--save_steps", default=500, type=int, help="")
    parser.add_argument("--log_steps", default=100, type=int, help="")
    parser.add_argument("--debug", default=False, action="store_true", help="")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="")
    parser.add_argument("--model_path", default="", type=str)
    parser.add_argument("--overwrite", default=False, action="store_true", help="")
    parser.add_argument("--seed", default=42, type=int, help="")
    parser.add_argument("--load_8bit", default=False, action="store_true", help="")
    parser.add_argument("--load_4bit", default=False, action="store_true", help="")
    parser.add_argument("--load_bf16", default=False, action="store_true", help="")
    args = parser.parse_args()
    print(args)
    return args


def set_seed(seed):
    rnd = random.Random()
    rnd.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return rnd


def get_model(model_path, load_8bit, load_4bit, load_bf16):
    device = 'cuda'
    if (Path(model_path) / 'adapter_config.json').exists():
        load_adapter = True
    else:
        load_adapter = False

    if load_adapter:
        print(datetime.datetime.now(), f'Loading Peft Model from {model_path} ...')
        peft_config = PeftConfig.from_pretrained(model_path)
        model, tokenizer = load_model_hf(peft_config.base_model_name_or_path, load_8bit, load_4bit, load_bf16, device=device)
        model = PeftModel.from_pretrained(model, model_path, device_map=device)
        processor = context_len = None
    else:
        print(datetime.datetime.now(), f'Loading Huggingface Model from {model_path} ...')
        model, tokenizer = load_model_hf(model_path, load_8bit, load_4bit, load_bf16, device=device)
        processor = context_len = None
    return tokenizer, model, processor, context_len


def compute_clm_loss(logits, labels, reduction, pad_id=0):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    vocab_size = logits.size(-1)
    loss_fct = CrossEntropyLoss(reduction=reduction)
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model/pipeline parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    loss = loss * (shift_labels != pad_id)
    return loss


def generate_prompt_qa(
        args,
        b,
        num_options,
        tokenizer,
        qa_mode='mcqa',
        append_option=False,
        option_j=None):
    assert qa_mode in ['mcqa', 'genqa']
    context_name = 'Transcript' if args.data_type != 'siqa' else 'Context'
    inp = ''
    inp += f'# {context_name} \n' + b['transcript'] + '\n\n'
    inp += '# Question \n' + b['q']
    if args.exclude_options_from_prompt:
        if append_option:
            opt = b[f'a{option_j}']
        system_prompt = 'You are an expert at understanding human communication. ' \
            'Please leverage the information provided and generate an answer in one sentence to the question.'
    else:
        inp += '\n\n' + '# Options \n'
        inp += '[A] ' + b['a0'] + '\n'
        if num_options >= 2:
            inp += '[B] ' + b['a1'] + '\n'
        if num_options >= 3:
            inp += '[C] ' + b['a2'] + '\n'
        if num_options >= 4:
            inp += '[D] ' + b['a3'] + '\n'
        if num_options >= 5:
            raise NotImplementedError(f'`num_options = {num_options}` is not supported yet')
        if append_option:
            opt = '[' + idx_to_option[option_j] + ']'
        system_prompt = 'You are an expert at understanding human communication. ' \
            'Please leverage the information provided and choose the most probable answer to the question from the options. ' \
            'Output your final answer by strictly following this format: [A], [B], [C], or [D]'
    if args.do_disable_system_prompt:
        # For llms that do not support system prompts, such as Mistral
        chat = [
            {"role": "user", "content": system_prompt + '\n\n' + inp}
        ]
    else:
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": inp}
        ]
    if qa_mode == 'mcqa' and append_option: # for evaluation
        chat.append({"role": "assistant", "content": opt})
    if qa_mode == 'genqa' and args.cot_prompt is not None:
        res_prefix = cot_prompts[args.cot_prompt]
        chat.append({"role": "assistant", "content": res_prefix})
    if not 'google/flan-t5' in args.model_path:
        if 'gemma' in args.model_path:
            prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=args.cot_prompt is None)
        else:
            prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    else:
        # For LLMs Without Multi-Turn Conversation Ability
        prompt = "\n\n".join([c["content"] for c in chat])
    if qa_mode == 'genqa' and args.cot_prompt is not None:
        if 'gemma' in args.model_path:
            if prompt.endswith("<end_of_turn>\n"):
                prompt = prompt[:-len("<end_of_turn>\n")].strip()
        else:
            if prompt.endswith(tokenizer.eos_token):
                prompt = prompt[:-len(tokenizer.eos_token)].strip()
    return prompt


def evaluate(args, model, processor, tokenizer, eval_examples, data_name, output_file):

    batch_size = args.eval_batch_size
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(model.config, "max_position_embeddings"):
        tokenizer.max_length = model.config.max_position_embeddings
    else:
        tokenizer.max_length = 8192

    if 'Llama-3' in args.model_path:
        conv_mode = "llama_3"
    else:
        conv_mode = "default"
    print(datetime.datetime.now(), f'`conv_mode` is set to {conv_mode}')

    n_examples = len(eval_examples)
    n_batches = n_examples // batch_size + int(n_examples % batch_size != 0)
    all_outputs = {}

    num_options = data_to_num_options[args.data_type]

    for i in tqdm(range(n_batches)):
        batch = eval_examples[i*batch_size:(i+1)*batch_size]

        if args.do_mc_qa:
            lls = []
            for j in range(num_options): # iterate for num of options
                inputs_text = []
                for b in batch:
                    prompt = generate_prompt_qa(
                        args, b, num_options, tokenizer, qa_mode='mcqa', append_option=True, option_j=j)
                    inputs_text.append(prompt)

                inputs = tokenizer(inputs_text, add_special_tokens=False, padding=True, truncation=True, return_tensors='pt')
                inputs = {k: inputs[k].cuda() for k in inputs}
                with torch.inference_mode():
                    outputs = model(**inputs)
                lls.append(outputs.loss.view(len(batch), -1).sum(1, keepdim=True).cpu())
            lls = torch.cat(lls, dim=1)
            if i % args.log_steps == 0:
                print(datetime.datetime.now())
                print('Input', json.dumps(batch[0], indent=4))
                print('Decoded input', tokenizer.decode(inputs['input_ids'][0]))
                print('Input IDs', inputs['input_ids'][0])
                print('Output (NLL)', lls[0])
            for b, ll in zip(batch, lls):
                ll = ll.cpu().numpy().tolist()
                all_outputs[b['q_id']] = {
                    'll': ll,
                    'pred': idx_to_option[int(np.argmin(ll))]
                }

        if args.do_gen_qa:
            if 'Llama-2' in args.model_path:
                response_template = "[/INST]"
                tokenizer.padding_side = "left"
            elif 'Llama-3' in args.model_path:
                response_template = "<|start_header_id|>assistant<|end_header_id|>"
                tokenizer.padding_side = "left"
            elif 'mistralai' in args.model_path:
                response_template = "[/INST]"
                tokenizer.padding_side = "left"
            elif 'flan-t5' in args.model_path:
                response_template = ""
                tokenizer.padding_side = "left"
            elif 'gemma' in args.model_path:
                response_template = "<start_of_turn>model"
                tokenizer.padding_side = "left"
            else:
                raise NotImplementedError(args.model_path)

            inputs_text = []
            for b in batch:
                prompt = generate_prompt_qa(
                    args, b, num_options, tokenizer, qa_mode='genqa')
                inputs_text.append(prompt)
            inputs = tokenizer(inputs_text, add_special_tokens=False, padding=True, truncation=True, return_tensors='pt')
            inputs = {k: inputs[k].cuda() for k in inputs}
            inputs["max_new_tokens"] = 256
            if args.do_sample:
                inputs["do_sample"] = True
                inputs["top_p"] = args.top_p
                inputs["temperature"] = args.temperature
            with torch.inference_mode():
                if conv_mode == 'llama_3':
                    outputs = model.generate(
                        **inputs,
                        eos_token_id=[
                            tokenizer.eos_token_id,
                            tokenizer.convert_tokens_to_ids("<|eot_id|>")
                        ],
                        pad_token_id=tokenizer.eos_token_id
                    )
                else:
                    outputs = model.generate(**inputs)
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            if i % args.log_steps == 0:
                print(datetime.datetime.now())
                print('Input', json.dumps(batch[0], indent=4))
                print('Decoded input', tokenizer.decode(inputs['input_ids'][0]))
                print('Input IDs', inputs['input_ids'][0])
                print('Decoded output', decoded_outputs[0])
            for b, o in zip(batch, decoded_outputs):
                if not b['q_id'] in all_outputs:
                    all_outputs[b['q_id']] = {}
                if response_template != "":
                    o = o.split(response_template)[-1].strip()
                all_outputs[b['q_id']]['gen'] = o

    if output_file.exists():
        results = load_json(output_file)
        print(datetime.datetime.now(), f'{str(output_file)} is loaded, and will be overwritten.')
    else:
        results = {}
    true_or_false = []
    true_or_false_gen = []
    for e in eval_examples:
        q_id = e['q_id']
        a_corr = e['a_str']
        a_idx = e['a_idx']
        if q_id in results:
            result = results[q_id]
        else:
            result = {}
        result['q_id'] = q_id
        result['a_str'] = a_corr
        result['a_true'] = idx_to_option[a_idx]
        if 'pred' in all_outputs[q_id]:
            result['a_pred'] = all_outputs[q_id]['pred']
            true_or_false.append(idx_to_option[a_idx] == all_outputs[q_id]['pred'])
        if 'll' in all_outputs[q_id]:
            result['a_score'] = all_outputs[q_id]['ll']
        if 'gen' in all_outputs[q_id]:
            result['a_gen'] = all_outputs[q_id]['gen']
            a_pred_gen = re.search(r'\[[A,B,C,D]\]', all_outputs[q_id]['gen'])
            if a_pred_gen is not None:
                start, end = a_pred_gen.span()
                a_pred_gen = all_outputs[q_id]['gen'][start:end][1]
            else:
                a_pred_gen = ''
            result['a_pred_gen'] = a_pred_gen
            true_or_false_gen.append(idx_to_option[a_idx] == a_pred_gen)
        if 'scenario_id' in e:
            result['scenario_id'] = e['scenario_id']
        results[q_id] = result
    if args.do_mc_qa:
        acc = float(np.mean(true_or_false))
        print('Accuracy:', acc)
    if args.do_gen_qa:
        acc = float(np.mean(true_or_false_gen))
        print('Accuracy:', acc)
    save_json(results, output_file)
    print(datetime.datetime.now(), f'Saved {str(output_file)}.')
    return results


def main():
    args = ArgParser()

    rnd = set_seed(args.seed)
    args.rnd = rnd

    # Disable system prompt for some llms
    args.do_disable_system_prompt = ('mistralai' in args.model_path or 'gemma' in args.model_path)

    output_dir = Path(args.output_dir) / args.exp_id
    output_dir.mkdir(parents=True, exist_ok=True)    

    for test_file in args.test_files:
        data_name = args.data_type + '_' + Path(test_file).stem # used for output file name
        output_name = f'predictions_on_{data_name}_{args.output_suffix}.json'
        output_file = output_dir / output_name
        if output_file.exists():
            print(datetime.datetime.now(), f'[WARNING] {str(output_file)} already exists.')
            if not args.overwrite:
                return None
        else:
            save_json({'CreatedAt': str(datetime.datetime.now())}, output_file)
        eval_examples = load_json(test_file)

        ## Load Model
        tokenizer, model, processor, context_len = get_model(
            args.model_path, args.load_8bit, args.load_4bit, args.load_bf16)
        model.config.use_cache = True

        if args.debug:
            eval_examples = eval_examples[:8]
        print(datetime.datetime.now(), f'{len(eval_examples)} examples will be evaluated.')
        results = evaluate(args, model, processor, tokenizer, eval_examples, data_name, output_file)


if __name__ == '__main__':
    main()
