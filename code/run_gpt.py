import os
import re
import sys
import json
import datetime
import base64
import time
import requests
import traceback
import argparse
from collections import defaultdict

import numpy as np
from pathlib import Path
import cv2
import webvtt
from tqdm import tqdm
import openai
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID"))

from src.utils import load_json, save_json
from prompts import cot_prompts

idx_to_option = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
}


def ArgParser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_type", default="siq", type=str, help="")
    parser.add_argument("--output_dir", default="output", type=str, help="")
    parser.add_argument("--do_gen_qa", default=False, action="store_true", help="")
    parser.add_argument("--do_mc_qa", default=False, action="store_true", help="")
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--exp_id", default="default", type=str)
    parser.add_argument("--test_file", default=None, type=str)
    parser.add_argument("--cot_prompt", default=None, type=str)
    parser.add_argument("--log_steps", default=10, type=int, help="")
    parser.add_argument("--seed", default=10, type=int, help="")
    parser.add_argument("--overwrite", default=False, action="store_true", help="")
    parser.add_argument("--debug", default=False, action="store_true", help="")
    args = parser.parse_args()
    print(args)
    return args

def main():
    args = ArgParser()

    # Load Dataset
    if args.data_type == 'tomato':
        output_dir = Path(args.output_dir) / args.exp_id
        output_dir.mkdir(parents=True, exist_ok=True)

        data_name = args.data_type + '_' + Path(args.test_file).stem
        output_name = f'predictions_on_{data_name}_.json'
        output_file = output_dir / output_name
        if output_file.exists():
            if not args.overwrite:
                print(f'[WARNING] {str(output_file)} already exists.')
                sys.exit()
            results = load_json(output_file)
            save_json(results, output_dir / ('old_' + output_name))
        else:
            results = {}
    elif args.data_type == 'siqa':
        output_dir = Path(args.output_dir) / args.exp_id
        output_dir.mkdir(parents=True, exist_ok=True)

        data_name = args.data_type + '_' + Path(args.test_file).stem
        output_name = f'predictions_on_{data_name}_.json'
        output_file = output_dir / output_name
        if output_file.exists():
            if not args.overwrite:
                print(f'[WARNING] {str(output_file)} already exists.')
                sys.exit()
            results = load_json(output_file)
            save_json(results, output_dir / ('old_' + output_name))
        else:
            results = {}

        data_id = args.data_type + '/' + Path(args.test_file).stem
        eval_examples = load_json(args.test_file)
        if args.debug:
            eval_examples = eval_examples[:20]
        num_options = 3

    # Multiple-choice QA
    if args.do_mc_qa:
        true_or_false = []
        errors = []

        step = 0
        for qa in tqdm(eval_examples):
            q_id = qa['q_id']
            if q_id in results:
                result = results[q_id]
            else:
                result = {}

            q = qa['q']
            a_str = qa['a_str']
            a_idx = qa['a_idx']

            result['a_str'] = a_str
            result['a_true'] = idx_to_option[a_idx]
            if 'scenario_id' in qa:
                result['scenario_id'] = qa['scenario_id']

            system_prompt = \
                "You are an expert at answering questions. " \
                "Please choose the most probable answer to the following question from the options. " \
                "Output your final verdict by strictly following this format: [A], [B], [C], or [D]"
            instruction = "# Question\n" + q + "\n\n"

            if num_options >= 1:
                instruction += "# Options\n[A] " + qa['a0']
            if num_options >= 2:
                instruction += "\n" + "[B] " + qa['a1']
            if num_options >= 3:
                instruction += "\n" + "[C] " + qa['a2']
            if num_options >= 4:
                instruction += "\n" + "[D] " + qa['a3']
            if args.cot_prompt is not None:
                instruction +=  "\n\n" + cot_prompts[args.cot_prompt]
            PROMPT_MESSAGES = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction},
            ]
            if step % args.log_steps == 0:
                print(PROMPT_MESSAGES)

            params = {
                "model": args.model,
                "messages": PROMPT_MESSAGES,
                "temperature": 0.6,
                "top_p": 0.9,
                "max_tokens": 500,
                "seed": args.seed,
            }
            if 'a_pred_gen' in result:
                print(datetime.datetime.now(), f'q_id: {q_id} has been chosen before, so it is skipped.')
            else:
                n_trial = 0
                max_trial = 10
                rest = 10
                while n_trial < max_trial:
                    try:
                        response = client.chat.completions.create(**params)
                        a_mc = response.choices[0].message.content
                        result['a_gen'] = a_mc
                        if step % args.log_steps == 0:
                            print(datetime.datetime.now(), f'Question {q_id} - Correct Option:')
                            print(idx_to_option[a_idx])
                            print(datetime.datetime.now(), f'Question {q_id} - Multiple-choice QA:')
                            print(a_mc)
                            print(datetime.datetime.now(), f'Question {q_id} - Multiple-choice QA: Suceeded!')
                        time.sleep(1)
                        break
                    except:
                        traceback.print_exc()
                        n_trial += 1
                        print(datetime.datetime.now(), f'Question {q_id} - Multiple-choice QA: ... failed {n_trial} times. Let me try again after {rest} seconds ...')
                        time.sleep(rest)
            if 'a_gen' in result:
                a_pred = extract_answer_from_response(result['a_gen'])
                result['a_pred_gen'] = a_pred
                true_or_false.append(result['a_true'] == a_pred)
            else:
                errors.append(q_id)

            step += 1
            results[q_id] = result
            save_json(results, output_file)

    save_json(results, output_file)
    print('Saved:', output_file)
    if args.do_mc_qa:
        print(datetime.datetime.now(), f'Num. of Errors: {len(errors)}')


def extract_answer_from_response(response):
    a_pred = re.search(r'\[[A,B,C,D]\]', response)
    if a_pred is not None:
        s, e = a_pred.span()
        a_pred = response[s:e][1]
        return a_pred
    else:
        return ''


if __name__ == '__main__':
    main()
