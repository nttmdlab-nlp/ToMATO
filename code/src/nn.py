import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoConfig, BitsAndBytesConfig
from src.utils import remove_tom


def load_model_hf(model_path, load_8bit, load_4bit, load_bf16, device='auto'):
    kwargs = {"device_map": "auto"}
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    elif load_bf16:
        kwargs['torch_dtype'] = torch.bfloat16
    else:
        kwargs['torch_dtype'] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    if 'google/flan-t5' in model_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    return model, tokenizer


def step(model, tokenizer, conv_mode, prompts, do_sample=False, top_p=0.9, temperature=0.6):
    inputs = tokenizer(prompts, add_special_tokens=True, return_tensors='pt', padding=True)
    for k in inputs:
        inputs[k] = inputs[k].to(model.device)
    inputs["max_new_tokens"] = 256
    if do_sample:
        inputs["do_sample"] = True
        inputs["top_p"] = top_p
        inputs["temperature"] = temperature
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
    responces = []
    for out in decoded_outputs:
        if conv_mode == 'llama_2':
            res = out.split('[/INST]')[-1].strip().replace("</s>", "")
        elif conv_mode == 'llama_3':
            res = out.split('<|start_header_id|>assistant<|end_header_id|>\n\n')[-1].strip().replace("<|eot_id|>", "")
        responces.append(res)
    return responces


def multi_turn_conversation(
        model,
        tokenizer,
        conv_mode,
        n_turn=0,
        agent1="",
        agent2="",
        system1="",
        system2="",
        init_inst1="",
        init_inst2="",
        tom_prompt1="",
        tom_prompt2="",
        inner_speech=False,
        do_sample=False,
        top_p=0.9,
        temperature=0.6,
        keep_inner_speech=False):
    conv1 = []
    conv2 = []

    conv1.append({'role': 'system', 'content': system1})
    conv2.append({'role': 'system', 'content': system2})

    if inner_speech and (not keep_inner_speech):
        init_inst2 = remove_tom(init_inst2)
    conv2.append({'role': 'user', 'content': init_inst2})
    conv2.append({'role': 'assistant', 'content': init_inst1})
    if inner_speech and (not keep_inner_speech):
        init_inst1 = remove_tom(init_inst1)
    conv1.append({'role': 'user', 'content': init_inst1})
    messages = []

    for i in range(n_turn):
        prompt1 = tokenizer.apply_chat_template(conv1, tokenize=False)
        if inner_speech:
            prompt1 += tom_prompt1
        res1 = step(model, tokenizer, conv_mode, [prompt1], do_sample=do_sample, top_p=top_p, temperature=temperature)[0]
        conv1.append({'role': 'assistant', 'content': res1})
        messages.append(['Agent A', res1])
        if inner_speech and (not keep_inner_speech):
            res1 = remove_tom(res1)
        conv2.append({'role': 'user', 'content': res1})
        prompt2 = tokenizer.apply_chat_template(conv2, tokenize=False)
        if inner_speech:
            prompt2 += tom_prompt2
        res2 = step(model, tokenizer, conv_mode, [prompt2], do_sample=do_sample, top_p=top_p, temperature=temperature)[0]
        conv2.append({'role': 'assistant', 'content': res2})
        messages.append(['Agent B', res2])
        if inner_speech and (not keep_inner_speech):
            res2 = remove_tom(res2)
        conv1.append({'role': 'user', 'content': res2})
    return conv1, conv2, messages
