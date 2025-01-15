import pathlib
from src.utils import get_tom_prompt, mental_verb, mental_states, save_json, load_json

sotopia_combo = load_json('sotopia/combo.json')
sotopia_agent = {a['pk']: a for a in load_json('sotopia/agent.json')}
sotopia_environment = {a['pk']: a for a in load_json('sotopia/environment.json')}

p_naive_prompt = {
    'Openness to Experience': 'an open',
    'Conscientiousness': 'conscientious',
    'Extraversion': 'extraversive',
    'Agreeableness': 'agreeable',
    'Neuroticism': 'neurotic',
}
p_naive_prompt_reversed = {
    'Openness to Experience': 'a closed',
    'Conscientiousness': 'unconscientious',
    'Extraversion': 'introversive',
    'Agreeableness': 'disagreeable',
    'Neuroticism': 'stable',
}


def big_five_to_description(big_five, p_prompt_type='naive'):
    if p_prompt_type == 'naive':
        p_description = 'You are '
        for i, p in enumerate(big_five.split(';')):
            key, value = p.strip().split(' - ')
            des_i = p_naive_prompt[key] if value == 'High' else p_naive_prompt_reversed[key]
            if i < 4:
                p_description += des_i + ', '
            else:
                p_description += 'and ' + des_i + ' person.'
    else:
        raise NotImplementedError(p_prompt_type)
    return p_description

def system_prompt_1st(agent, other, env, mental_state, i):
    pronoun = other["gender_pronoun"].lower().split("/")
    system = f'Your name is {agent["first_name"]} {agent["last_name"]}, a {agent["age"]}-year-old {agent["occupation"]}.\n' \
    f'You are talking with {other["first_name"]} {other["last_name"]}, a {other["age"]}-year-old {other["occupation"]}.\n' \
    f'The scenario of this conversation: {env["scenario"]}.\nYour goal: {env["agent_goals"][i]}\n' \
    f'Your personality: {big_five_to_description(agent["big_five"], p_prompt_type="naive")}\n' \
    f'Please have a conversation with {pronoun[1]} while thinking about your {mental_state} from ( to ) in one sentence.\n' \
    'Please generate different thoughts and utterances in different turns.\n' \
    f'After thinking about your {mental_state} briefly, please finish your thought with ) and speak to {pronoun[1]} briefly in one or two sentences based on your thought.\n' \
    'Output your thought and utterance by strictly following this format: (your thought) "your utterance".'
    return system

def system_prompt_2nd(agent, other, env, mental_state, i):
    pronoun = other["gender_pronoun"].lower().split("/")
    system = f'Your name is {agent["first_name"]} {agent["last_name"]}, a {agent["age"]}-year-old {agent["occupation"]}.\n' \
    f'You are talking with {other["first_name"]} {other["last_name"]}, a {other["age"]}-year-old {other["occupation"]}.\n' \
    f'The scenario of this conversation: {env["scenario"]}.\nYour goal: {env["agent_goals"][i]}\n' \
    f'Your personality: {big_five_to_description(agent["big_five"], p_prompt_type="naive")}\n' \
    f'Please have a conversation with {pronoun[1]} while thinking about {other["first_name"]}\'s {mental_state} from ( to ) in one sentence.\n' \
    'Please generate different thoughts and utterances in different turns.\n' \
    f'After thinking about {other["first_name"]}\'s {mental_state} briefly, please finish your thought with ) and speak to {pronoun[1]} briefly in one or two sentences based on your thought.\n' \
    'Output your thought and utterance by strictly following this format: (your thought) "your utterance".'
    return system

def main():
    Path('data/scenarios/').mkdir(exist_ok=True, parents=True)
    test_ids = load_json('data/test_ids.json')

    for i, combo in enumerate(sotopia_combo):
        combo_id = combo['pk']
        if not combo_id[-5:] in test_ids:
            continue
        for mental_state in mental_states:
            scenario = {}
            scenario['sotopia'] = {}
            scenario['sotopia']['combo_id'] = combo_id
            scenario['sotopia']['env_id'] = combo['env_id']
            scenario['sotopia']['agent_ids'] = combo['agent_ids']

            env = sotopia_environment[combo['env_id']]
            agent1 = sotopia_agent[combo['agent_ids'][0]]
            agent2 = sotopia_agent[combo['agent_ids'][1]]
            pronoun1 = agent1["gender_pronoun"].lower().split("/")
            pronoun2 = agent2["gender_pronoun"].lower().split("/")

            system1 = system_prompt_1st(agent1, agent2, env, mental_state, 0)
            tom_order1 = 1

            system2 = system_prompt_2nd(agent2, agent1, env, mental_state, 1)
            tom_order2 = 2

            scenario['init_inst1'] = '() "Hi, how are you?"'
            scenario['init_inst2'] = '() "Hi!"'

            tom_prompt1 = get_tom_prompt(mental_state, tom_order1, pronoun=pronoun2[0])
            tom_prompt2 = get_tom_prompt(mental_state, tom_order2, pronoun=pronoun1[0])

            scenario['system1'] = system1
            scenario['system2'] = system2
            scenario['tom_prompt1'] = tom_prompt1
            scenario['tom_prompt2'] = tom_prompt2
            scenario_name = f'{mental_state}_{combo_id[-5:]}'
            save_json(scenario, f'data/scenarios/{scenario_name}.json')

    Path('assignments/').mkdir(parents=True, exist_ok=True)
    d = Path('assignments/')
    order = 1.5

    gpu_id = 0
    fs = {}
    fs[0] = open(d / 'gpu0.txt', 'w')
    # fs[1] = open(d / 'gpu1.txt', 'w')
    # fs[2] = open(d / 'gpu2.txt', 'w')
    # fs[3] = open(d / 'gpu3.txt', 'w')

    n = 0

    for i, combo in enumerate(sotopia_combo):
        for mental_state in mental_states:
            combo_id = combo['pk']
            scenario_name = f'{mental_state}_{combo_id[-5:]}'
            fs[gpu_id].write(scenario_name + "\n")
            n += 1
        gpu_id = (gpu_id + 1) % len(fs)
    for i in fs:
        fs[i].close()
    print(n, 'scenarios in total')

if __name__ == '__main__':
    main()
