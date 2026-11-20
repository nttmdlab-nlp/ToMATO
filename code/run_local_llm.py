import argparse
import json
import logging
import re
from pathlib import Path
from vllm import LLM, SamplingParams

from prompts import cot_prompts


LOGGER = logging.getLogger(__name__)
DATA_TO_NUM_OPTIONS = {"tomato": 4, "siqa": 3}
OPTIONS = "ABCD"
DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate local chat models with vLLM offline inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_files", nargs="+", type=Path, required=True)
    parser.add_argument("--data_type", choices=DATA_TO_NUM_OPTIONS, default="tomato")
    parser.add_argument("--torch_dtype", default="auto", choices=(
        "auto", "float16", "bfloat16", "float32",
    ), help="Passed to vLLM as dtype; quantization is a separate setting.")
    parser.add_argument("--quantization", default=None,
                        help="vLLM quantization method, e.g. awq or gptq; auto-detected if omitted.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--max_num_seqs", type=int, default=256,
                        help="Maximum concurrent sequences scheduled by vLLM.")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--disable_system_prompt", "--do_disable_system_prompt",
                        action="store_true",
                        help="Merge instructions into the user message for templates without system roles.")
    parser.add_argument("--exclude_options_from_prompt", action="store_true")
    parser.add_argument("--cot_prompt", choices=cot_prompts, default=None)
    parser.add_argument("--do_gen_qa", action="store_true",
                        help="Compatibility flag; generation-based evaluation is always enabled.")
    parser.add_argument("--do_mc_qa", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=Path("output"))
    parser.add_argument("--exp_id", default="default")
    parser.add_argument("--output_suffix", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Evaluate only the first 8 examples.")
    args = parser.parse_args(argv)
    if args.do_mc_qa:
        parser.error("--do_mc_qa is no longer supported; use generation-based evaluation (--do_gen_qa).")
    for name in ("tensor_parallel_size", "max_num_seqs", "max_new_tokens"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.max_model_len is not None and args.max_model_len <= 0:
        parser.error("--max_model_len must be positive")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu_memory_utilization must be between 0 and 1")
    if not 0 < args.top_p <= 1:
        parser.error("--top_p must be in (0, 1]")
    if not 0 <= args.temperature < float("inf"):
        parser.error("--temperature must be finite and nonnegative")
    return args


def build_messages(args, example):
    context_name = "Context" if args.data_type == "siqa" else "Conversation"
    content = f"# {context_name}\n{example['conversation']}\n\n# Question\n{example['q']}"
    instruction = "You are an expert at understanding human communication. "
    if args.exclude_options_from_prompt:
        instruction += (
            "Please leverage the information provided and generate an answer "
            "in one sentence to the question."
        )
    else:
        labels = OPTIONS[:DATA_TO_NUM_OPTIONS[args.data_type]]
        choices = "\n".join(
            f"[{label}] {example[f'a{index}']}" for index, label in enumerate(labels)
        )
        content += f"\n\n# Options\n{choices}"
        if args.cot_prompt is not None:
            instruction += (
                "Please leverage the information provided and choose the most probable "
                "answer to the question from the options: " + ", ".join(f"[{label}]" for label in labels) + ". "
                "Reason step by step before outputting your final answer. "
                "After providing your explanation, output your final verdict by strictly following this format: "
                "[[A]] if A is better, [[B]] if B is better, [[C]] if C is better, and [[D]] if D is better."
            )
        else:
            instruction += (
                "Please leverage the information provided and choose the most probable "
                "answer to the question from the options. Output your final answer by "
                "strictly following this format: [[A]] if A is better, [[B]] if B is better, [[C]] if C is better, and [[D]] if D is better."
            )
    if args.cot_prompt is not None:
        content += "\n\n" + cot_prompts[args.cot_prompt]
    if args.disable_system_prompt:
        return [{"role": "user", "content": instruction + "\n\n" + content}]
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": content},
    ]


def extract_answer(text, num_options):
    matches = re.findall(r"\[\[([" + OPTIONS[:num_options] + r"])\]\]", text)
    return matches[-1] if matches else ""


def load_examples(path, args):
    with path.open(encoding="utf-8") as handle:
        examples = json.load(handle)
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"{path}: expected a nonempty JSON list of examples")
    if args.debug:
        examples = examples[:8]
    seen = set()
    num_options = DATA_TO_NUM_OPTIONS[args.data_type]
    required = {"q_id", "conversation", "q", "a_str", "a_idx"}
    if not args.exclude_options_from_prompt:
        required.update(f"a{index}" for index in range(num_options))
    for index, example in enumerate(examples):
        if not isinstance(example, dict) or not required.issubset(example):
            raise ValueError(f"{path}: example {index} is missing required fields: {sorted(required)}")
        question_id = example["q_id"]
        if not isinstance(question_id, str) or not question_id or question_id in seen:
            raise ValueError(f"{path}: invalid or duplicate q_id: {question_id!r}")
        seen.add(question_id)
        if type(example["a_idx"]) is not int or not 0 <= example["a_idx"] < num_options:
            raise ValueError(f"{path}: invalid a_idx for {question_id}")
    return examples


def create_engine(args):
    options = {
        "model": args.model_path,
        "dtype": args.torch_dtype,
        "quantization": args.quantization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "trust_remote_code": args.trust_remote_code,
        "seed": args.seed,
    }
    if args.max_model_len is not None:
        options["max_model_len"] = args.max_model_len
    return LLM(**options)


def create_sampling_params(args):
    return SamplingParams(
        temperature=args.temperature if args.do_sample else 0.0,
        top_p=args.top_p if args.do_sample else 1.0,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )


def evaluate(args, engine, sampling_params, examples, output_file):
    conversations = [build_messages(args, example) for example in examples]
    outputs = engine.chat(conversations, sampling_params=sampling_params, use_tqdm=True)
    if len(outputs) != len(examples):
        raise RuntimeError("vLLM returned a different number of outputs than inputs")
    completions = {
        example["q_id"]: {
            "text": output.outputs[0].text,
            "finish_reason": output.outputs[0].finish_reason,
        }
        for example, output in zip(examples, outputs)
    }
    return save_results(args, examples, completions, output_file)


def summarize_results(results, compute_accuracy):
    total = len(results)
    correct = sum(result["a_pred_gen"] == result["a_true"] for result in results)
    return {
        "accuracy": correct / total if compute_accuracy and total else None,
        "correct": correct if compute_accuracy else None,
        "total": total,
        "invalid_answers": (sum(not result["a_pred_gen"] for result in results)
                            if compute_accuracy else None),
        "length_limited_responses": sum(result["finish_reason"] == "length"
                                       for result in results),
    }


def save_results(args, examples, completions, output_file):
    results = {}
    num_options = DATA_TO_NUM_OPTIONS[args.data_type]
    correct = invalid = truncated = 0
    for example in examples:
        completion = completions[example["q_id"]]
        answer = extract_answer(completion["text"], num_options)
        expected = OPTIONS[example["a_idx"]]
        result = {
            "q_id": example["q_id"],
            "a_str": example["a_str"],
            "a_true": expected,
            "a_gen": completion["text"],
            "a_pred_gen": answer,
            "finish_reason": completion["finish_reason"],
        }
        if "scenario_id" in example:
            result["scenario_id"] = example["scenario_id"]
        results[example["q_id"]] = result
        correct += answer == expected
        invalid += not answer
        truncated += completion["finish_reason"] == "length"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
    with temporary_file.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=4, ensure_ascii=False)
        handle.write("\n")
    temporary_file.replace(output_file)
    compute_accuracy = not args.exclude_options_from_prompt
    metrics = {
        "overall": summarize_results(list(results.values()), compute_accuracy),
        "subsets": {},
        "mental_states": {},
    }
    if args.data_type == "tomato":
        for subset in ("tomato_fb", "tomato_first", "tomato_second"):
            with (DATASET_DIR / f"{subset}.json").open(encoding="utf-8") as handle:
                question_ids = {example["q_id"] for example in json.load(handle)}
            metrics["subsets"][subset] = summarize_results(
                [result for question_id, result in results.items() if question_id in question_ids],
                compute_accuracy,
            )
        with (DATASET_DIR / "tomato.json").open(encoding="utf-8") as handle:
            all_examples = json.load(handle)
        mental_states = {}
        for example in all_examples:
            mental_states.setdefault(example["mental_state"], set()).add(example["q_id"])
        for mental_state, question_ids in sorted(mental_states.items()):
            metrics["mental_states"][mental_state] = summarize_results(
                [result for question_id, result in results.items() if question_id in question_ids],
                compute_accuracy,
            )
    evaluation_file = output_file.parent / "eval_results.json"
    evaluation_results = {}
    if evaluation_file.exists():
        with evaluation_file.open(encoding="utf-8") as handle:
            evaluation_results = json.load(handle)
    evaluation_results[output_file.name] = metrics
    temporary_evaluation_file = evaluation_file.with_suffix(".json.tmp")
    with temporary_evaluation_file.open("w", encoding="utf-8") as handle:
        json.dump(evaluation_results, handle, indent=4, ensure_ascii=False)
        handle.write("\n")
    temporary_evaluation_file.replace(evaluation_file)
    LOGGER.info("Saved evaluation metrics: %s", evaluation_file)
    if args.exclude_options_from_prompt:
        LOGGER.info("Free-form generation: automatic choice accuracy is not computed.")
    else:
        LOGGER.info("Accuracy: %.4f (%d/%d); invalid answers: %d",
                    correct / len(examples), correct, len(examples), invalid)
    LOGGER.info("Length-limited responses: %d; saved %s", truncated, output_file)
    return results


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    output_dir = args.output_dir / args.exp_id
    jobs = []
    output_paths = set()
    for test_file in args.test_files:
        output_file = output_dir / f"predictions_on_{test_file.stem}_{args.output_suffix}.json"
        if output_file in output_paths:
            raise ValueError(f"Multiple test files map to the same output: {output_file}")
        output_paths.add(output_file)
        if output_file.exists() and not args.overwrite:
            LOGGER.info("Skipping existing output: %s", output_file)
            continue
        jobs.append((load_examples(test_file, args), output_file))
    if not jobs:
        return
    engine = create_engine(args)
    sampling_params = create_sampling_params(args)
    for examples, output_file in jobs:
        LOGGER.info("Evaluating %d examples -> %s", len(examples), output_file)
        evaluate(args, engine, sampling_params, examples, output_file)


if __name__ == "__main__":
    main()
