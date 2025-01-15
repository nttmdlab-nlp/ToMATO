# ToMATO Benchmark
This directory contains the ToMATO benchmark, which we presented in our paper.<br>

## Contents
- tomato.json
  - contains all the examples in ToMATO
- tomato_first.json
  - contains first-order ToM questions in ToMATO
- tomato_second.json
  - contains second-order ToM questions in ToMATO
- tomato_fb.json
  - contains false-belief tasks in ToMATO (ToMATO-FB)

## Data Format
```
{
    "a0": <option 0>,
    "a1": <option 1>,
    "a2": <option 2>,
    "a3": <option 3>,
    "a_idx": <correct option id>,
    "a_str": <correct option>,
    "big_five": <the personality trait of target speaker whose mental state is being asked>,
    "conversation": <conversation>,
    "false_belief": <true or false>,
    "mental_state": <mental state>,
    "order": <the order of the asked mental state>,
    "q": <question>,
    "q_id": <question id>,
    "sotopia_agents": [
        <agent id 1>,
        <agent id 2>
    ],
    "sotopia_environment": <environment id>,
    "u_id": <utterance id>
},
```

## License
Since the examples are generated with Llama-3, the dataset is released under META LLAMA 3 COMMUNITY LICENSE.
