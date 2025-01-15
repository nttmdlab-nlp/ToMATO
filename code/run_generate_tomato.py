import os
import re
import sys
import json
import random
import datetime
import base64
import time
import requests
import traceback
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from src.utils import (
    load_json, save_json, remove_tom, get_formatted_conv, is_match_inner_speech_format, separate_tom,
    mental_states, mental_verb, change_pronoun
)


def ArgParser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_path", default="meta-llama/Meta-Llama-3-70B-Instruct", type=str)
    parser.add_argument("--output_dir", default="data/tomato", type=str, help="")
    parser.add_argument("--overwrite", default=False, action="store_true", help="")
    parser.add_argument("--max_utterances", default=20, type=int)
    args = parser.parse_args()
    print(args)
    return args


def convert_messages_to_examples(messages, scenario, scenario_id, sotopia_agent, tokenizer):
    agent_ids = scenario['sotopia']['agent_ids']
    agents = [sotopia_agent[i] for i in agent_ids]
    agents = [f'{a["first_name"]} {a["last_name"]}' for a in agents]

    is_valid_format = True
    n_error = 0
    for message in messages['messages']:
        res = message[1]
        res = res.replace('\n', ' ')
        res = re.sub(r'[ ]+', ' ', res).strip()
        if not is_match_inner_speech_format(res):
            n_error += 1
            if n_error % 100 == 0:
                print(datetime.datetime.now(), f'Does not match Inner Speech format in {scenario_id}:', res)
            is_valid_format = False
    if not is_valid_format:
        return []

    all_messages = []
    all_tom = defaultdict(list)
    for message in messages['messages']:
        agent = message[0]
        m = message[1]
        m = m.replace('\n', ' ')
        m = re.sub(r'[ ]+', ' ', m).strip()
        tom, res = separate_tom(m)
        agent_id = agent_ids[0] if agent == 'Agent A' else agent_ids[1]
        all_messages.append([agent_id, tom, res])
        all_tom[agent_id].append(tom)

    valid_agent_ids = []
    for agent_id in all_tom:
        all_tom[agent_id] = list(set(all_tom[agent_id]))
        if len(all_tom[agent_id]) >= 4:
            valid_agent_ids.append(agent_id)
        else:
            n_error += 1
            if n_error % 100 == 0:
                print(datetime.datetime.now(), f'Num. of unique options is less than 4 in {scenario_id}:', len(all_tom[agent_id]))

    current_utterances = ''
    current_utterances += f'{agents[0]}: "Hi!"\n'
    current_utterances += f'{agents[1]}: "Hi, how are you?"\n'
    for j, (agent_id, tom, res) in enumerate(all_messages):
        agent = sotopia_agent[agent_id]
        agent_name = f'{agent["first_name"]} {agent["last_name"]}'
        current_utterances += f'{agent_name}: ' + res + '\n'

    mental_state = scenario_id.split('_')[1]
    verb = mental_verb[mental_state][0]
    third = mental_verb[mental_state][1]

    examples = []
    for j, (agent_id, tom, res) in enumerate(all_messages):
        ex = {}
        agent = sotopia_agent[agent_id]
        agent_name = f'{agent["first_name"]} {agent["last_name"]}'
        other_id = agent_ids[(agent_ids.index(agent_id) + 1) % 2]
        other = sotopia_agent[other_id]
        other_name = f'{other["first_name"]} {other["last_name"]}'

        if not agent_id in valid_agent_ids:
            continue

        ## Incorrect option selection
        toms = all_tom[agent_id]
        incorr = random.sample(list(filter(lambda t: t != tom, toms)), 3)
        if not incorr:
            print('[IncorrectOptionError]', 'Correct:', tom, ', Candidates:', toms)
            continue
        corr_idx = random.sample([0, 1, 2, 3], 1)[0]
        options = incorr[:corr_idx] + [tom] + incorr[corr_idx:]
        options = [o[1:-1] for o in options]

        pronoun = sotopia_agent[agent_id]['gender_pronoun'].split('/')[0]
        pronoun_other = sotopia_agent[other_id]['gender_pronoun'].split('/')[0]
        if j % 2 == 0:
            l = len(f'I {verb}')
            options = [f'{pronoun} {third}' + o[l:] for o in options]
            q_order = 1
        elif j % 2 == 1:
            l = len(f'I think that {pronoun_other.lower()} {third}')
            options = [f'{pronoun} thinks that {pronoun_other.lower()} {third}' + o[l:] for o in options]
            q_order = 2
        options = [change_pronoun(o, pronoun.lower(), pronoun_other.lower()) for o in options]

        ex['a_idx'] = corr_idx
        ex['a_str'] = options[corr_idx]
        for i in range(4):
            ex[f'a{i}'] = options[i]
        q = generate_question(mental_state, verb, third, q_order, agent["first_name"], other["first_name"],
            res, pronoun)
        ex['q'] = q
        ex['q_id'] = f'{scenario_id}-{j}'
        ex['transcript'] = current_utterances
        ex['scenario_id'] = scenario_id
        examples.append(ex)
    return examples


def generate_question(mental_state, verb, third, order, name1, name2,
        res, pronoun):
    assert order in [1, 2]
    pronoun = pronoun.lower()
    if mental_state == 'emotion':
        if order == 1:
            q = random.choice([
                f'How does {name1} {verb} when {pronoun} says {res}?',
                f'When {name1} says {res}, how does {pronoun} {verb}?'
            ])
        elif order == 2:
            q = random.choice([
                f'How does {name1} think that {name2} {third} when {pronoun} says {res}?',
                f'When {name1} says {res}, how does {pronoun} think that {name2} {third}?'
            ])
    elif mental_state == 'intention':
        if order == 1:
            q = random.choice([
                f'What will {name1} do when {pronoun} says {res}?',
                f'When {name1} says {res}, what will {name1} do?'
            ])
        elif order == 2:
            q = random.choice([
                f'What does {name1} think that {name2} will do when {pronoun} says {res}?',
                f'When {name1} says {res}, what does {name1} think that {name2} will do?'
            ])
    else:
        if order == 1:
            q = random.choice([
                f'What does {name1} {verb} when {pronoun} says {res}?',
                f'When {name1} says {res}, what does {name1} {verb}?'
            ])
        elif order == 2:
            q = random.choice([
                f'What does {name1} think that {name2} {third} when {pronoun} says {res}?',
                f'When {name1} says {res}, what does {name1} think that {name2} {third}?'
            ])
    return q


def main():
    args = ArgParser()

    args.model_name = args.model_path.split('/')[1]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'tomato.json'
    if not args.overwrite:
        if output_file.exists():
            print(datetime.datetime.now(), str(output_file), 'already exists. Please use `--overwrite` option to overwrite the file.')
            sys.exit()

    data_dir = Path(f'data/conversations/{args.model_name}')
    data = list(data_dir.glob('*.json'))

    sotopia_agent = {a['pk']: a for a in load_json('sotopia/agent.json')}

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, padding_side='left')

    random.seed(42)

    all_examples = []

    for i in tqdm(range(len(data))):
        scenario_id = data[i].stem
        messages = load_json(data[i])
        scenario = load_json(Path('data/scenarios/') / (scenario_id + '.json'))
        messages['messages'] = messages['messages'][:args.max_utterances]
        examples = convert_messages_to_examples(messages, scenario, scenario_id, sotopia_agent, tokenizer)
        all_examples.extend(examples)

    print(datetime.datetime.now(), 'Num. of Examples:', len(all_examples))
    save_json(all_examples, output_file)
    print('Saved:', str(output_file))


if __name__ == '__main__':
    main()
