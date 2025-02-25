# Codes for Reproduction
## Setup
### Software
The codes are tested with the following libraries.
Please update them if needed.

* transformers==4.43.2
* torch==2.3.0
* bitsandbytes==0.43.1
* openai==1.3.6

Other required libraries are listed in `requirements.txt`.
To install them, plese prepare a virtual environment if needed and then execute
```
pip install -r requirements.txt
```

## Generate ToMATO
### 0. Download SOTOPIA dataset
Please first download SOTOPIA environments, agents, and combo following [this url](https://github.com/sotopia-lab/sotopia).

First, set up a virtual environment for loading SOTOPIA following [this URL](https://docs.sotopia.world/#installation).
Second, follow the instructions in [this URL](https://github.com/sotopia-lab/sotopia/issues/7#issuecomment-1806365778) to get the raw SOTOPIA dataset and launch a docker container.
Then, convert the raw data into json formats as follows, while launching the docker container.

```
from sotopia.database.persistent_profile import AgentProfile, EnvironmentProfile
from sotopia.database.env_agent_combo_storage import EnvAgentComboStorage

def save_json(data, file):
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

agent_pks = AgentProfile.all_pks()
agent_pks = list(agent_pks)
agents = []
for pk in agent_pks:
    agents.append(AgentProfile.get(pk=pk))
output_agents = [a.__dict__ for a in agents]
save_json(output_agents, './data/sotopia/agent.json')

environment_pks = EnvironmentProfile.all_pks()
environment_pks = list(environment_pks)
print(len(environment_pks))
environments = []
for pk in environment_pks:
    environments.append(EnvironmentProfile.get(pk=pk))
output_environments = [e.__dict__ for e in environments]
save_json(output_environments, './data/sotopia/environment.json')

all_combos = EnvAgentComboStorage().all_pks()
all_combos = list(all_combos)
combos = []
for combo in all_combos:
    combos.append(EnvAgentComboStorage().get(combo))
output_combos = [c.__dict__ for c in combos]
save_json(output_combos, './data/sotopia/combo.json')
```

### 1. Generate System Prompts
```
python run_generate_system.py
```
This command will generate `data/scenarios/{scenario_name}.json`

### 2. LLM-LLM Conversations
```
GPU_ID="0"
N_TURN="7"
OUTPUT_DIR="data/conversations/"
SCENARIO_DIR="scenarios/"
SCENARIO_LIST="assignments/gpu0.txt"
MODEL_PATH="meta-llama/Meta-Llama-3-70B-Instruct"
IFS="/"; MODEL_NAME=($MODEL_PATH)
EXP_ID="Conv_${MODEL_NAME[1]}_gpu${GPU_ID}"
IFS=" "

CUDA_VISIBLE_DEVICES=$GPU_ID python -u run_inner_speech.py --exp_id $EXP_ID \
--scenario_dir $SCENARIO_DIR --scenario_list $SCENARIO_LIST --output_dir $OUTPUT_DIR --n_turn $N_TURN \
--model_path $MODEL_PATH --conv_mode llama_3 --do_sample --load_4bit
```
This command will generate `data/conversations/{scenario_name}.json`

### 3. Fromat ToMATO
```
OUTPUT_DIR="data/tomato"
MODEL_PATH="meta-llama/Meta-Llama-3-70B-Instruct"

python run_generate_tomato.py --output_dir $OUTPUT_DIR \
--model_path $MODEL_PATH --max_utterances 14 --overwrite
```
This command will generate `data/tomato/tomato.json`

## Evaluate LLMs on ToMATO
The following commands are for evaluating local and proprietary LLMs on ToMATO.

### Local LLM
```
GPU_IDS="0"
MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
EVAL_BATCH_SIZE="2"
TEST_FILES="data/tomato/tomato.json"
SEED="42"
IFS="/"; MODEL_NAME=($MODEL_PATH); METHOD=${MODEL_NAME[-1]}; IFS=" "
EXP_ID="${METHOD}_${SEED}"

CUDA_VISIBLE_DEVICES=$GPU_IDS python \
run_local_llm.py --do_eval --output_dir results/${EXP_ID} --exp_id $EXP_ID \
--model_path $MODEL_PATH --load_4bit --eval_batch_size $EVAL_BATCH_SIZE \
--test_files $TEST_FILES \
--data_type tomato --do_gen_qa --do_sample --seed $SEED
```

### GPT
```
FILE="data/tomato/tomato.json"
MODEL="gpt-4o-mini-2024-07-18"
OUTPUT_DIR="retults/${MODEL}"
SEED="42"
IFS="/"; F=($FILE); VERSION=${F[2]}; IFS=" "
EXP_ID="${MODEL}_${SEED}"

python run_gpt.py --data_type tomato \
--test_file $FILE --output_dir $OUTPUT_DIR --log_steps 200 \
--model $MODEL --exp_id $EXP_ID --do_mc_qa --seed $SEED
```

## License
The codes are released under NTT Licence, which allows them to be used for research purposes only.
