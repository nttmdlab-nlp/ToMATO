import re
import spacy
nlp = spacy.load('en_core_web_sm')
import json

def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)

def save_json(data, file):
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

def remove_tom(res):
    match = re.findall(r'\(.*\)', res)
    for m in match:
        res = res.replace(m, '')
    return res.strip()

def get_formatted_conv(messages):
    out = ''
    for m in messages:
        out += '> ' + m[0] + '\n'
        out += m[1] + '\n\n'
    return out

def is_match_inner_speech_format(res):
    match = re.fullmatch(r'\([^\(\)]*\) "[^"\(\)]*"', res)
    if match is not None:
        return True
    else:
        return False

def separate_tom(res):
    match = re.search(r'\(.*\)', res)
    s, e = match.span()
    tom = res[s:e].strip()
    res = res.replace(tom, '').strip()
    return tom, res

mental_states = ["emotion", "belief", "intention", "desire", "knowledge"]
mental_verb = {
    "emotion": ["feel", "feels", "felt"],
    "belief": ["think", "thinks", "thought"],
    "intention": ["will", "will", "would"],
    "desire": ["want", "wants", "wanted"],
    "knowledge": ["know", "knows", "knew"],
}

def get_tom_prompt(mental_state, order, pronoun=None):
    if order == 1:
        return f'(I {mental_verb[mental_state][0]}'
    elif order == 2:
        assert pronoun is not None
        return f'(I think that {pronoun} {mental_verb[mental_state][1]}'
    elif order == 0:
        return ''
    else:
        raise NotImplementedError(order)

def change_pronoun(input, pronoun, pronoun_other):
    assert pronoun in ['he', 'she', 'they']
    assert pronoun_other in ['he', 'she', 'they']
    if pronoun == 'he':
        dict = {'I': 'he', 'me': 'him', 'my': 'his', 'mine': 'his', 'myself': 'himself',
            'we': 'they', 'our': 'their', 'us': 'them', 'ours': 'theirs'}
        replace_dict = {'I\'m': 'he\'s', 'I\'ve': 'he has', 'I\'ll': 'he will', 'I\'d': 'he would'}
    elif pronoun == 'she':
        dict = {'I': 'she', 'me': 'her', 'my': 'her', 'mine': 'hers', 'myself': 'herself',
            'we': 'they', 'our': 'their', 'us': 'them', 'ours': 'theirs'}
        replace_dict = {'I\'m': 'she\'s', 'I\'ve': 'she has', 'I\'ll': 'she will', 'I\'d': 'she would'}
    elif pronoun == 'they':
        dict = {'I': 'they', 'me': 'them', 'my': 'their', 'mine': 'theirs', 'myself': 'themselves', 'I\'m': 'they\'re',
            'we': 'they', 'our': 'their', 'us': 'them', 'ours': 'theirs'}
        replace_dict = {'I\'m': 'they\'re', 'I\'ve': 'they have', 'I\'ll': 'they will', 'I\'d': 'they would'}

    if pronoun_other == 'he':
        dict.update({'you': 'he', 'you': 'him', 'your': 'his', 'yours': 'his', 'yourself': 'himself'})
        replace_dict.update({'you\'re': 'he\'s', 'you\'ve': 'he has', 'you\'ll': 'he will'})
    elif pronoun_other == 'she':
        dict.update({'you': 'she', 'you': 'her', 'your': 'her', 'yours': 'hers', 'yourself': 'herself'})
        replace_dict.update({'you\'re': 'she\'s', 'you\'ve': 'she has', 'you\'ll': 'she will'})
    elif pronoun_other == 'they':
        dict.update({'you': 'they', 'you': 'them', 'your': 'their', 'yours': 'theirs', 'yourself': 'themselves'})
        replace_dict.update({'you\'re': 'they\'re', 'you\'ve': 'they have', 'you\'ll': 'they will'})

    for k, v in replace_dict.items():
        input = input.replace(k, v)
    out  = []
    doc = nlp(input)
    for sent in doc.sents:
        for tok in sent:
            out.append(dict.get(tok.text, tok.text))
            out.append(tok.whitespace_)
        break
    return "".join(out)
