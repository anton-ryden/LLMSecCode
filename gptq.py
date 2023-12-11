import json
import time
import copy
import torch
from typing import List, Union, Tuple, Dict
from transformers.generation import GenerationConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from argument import parse_args, get_chat_template
from get_prompts import get_prompts
from test_code import test_code


def json_to_file(data: dict, json_path: str) -> None:
    # Write JSON into a file
    with open(json_path, "w") as json_file:
        json.dump(data, json_file, indent=6)


def format_code_response(llm_response: str, tokens: List[str]) -> str:
    if args.template_set == "llama":
        s_index = llm_response.find("```")
        e_index = llm_response.rfind("```")
        if s_index is not None and e_index is not None:
            llm_response = llm_response[s_index+3:e_index]

    # Remove specified tokens and format the response
    for token in tokens:
        llm_response = llm_response.replace(token, "")

    # Remove whitespace
    llm_response = llm_response.replace("\t", "    ")
    llm_response = llm_response.lstrip("\n")
    llm_response = llm_response.lstrip()
    return llm_response


def format_response(
    time_to_complete: float,
    answers: List[str],
    tokenizer: PreTrainedTokenizer,
    tokenized_chat: str,
    prompt: List[Dict],
) -> dict:
    tokens_generated = 0
    json_patches = []

    for ans in answers:
        # Remove instructions from response
        answ_no_ins = ans.replace(tokenized_chat, "")
        answ_no_ins = answ_no_ins[: len(answ_no_ins) - len(tokenizer.eos_token)]

        # Format response and calculate tokens/s
        formatted_code = format_code_response(
            answ_no_ins, [tokenizer.eos_token, tokenizer.bos_token]
        )

        # Create conversation json
        conv = copy.copy(prompt)
        conv.append({"role": "assistant", "content": formatted_code})

        # Check if code is correct
        is_code_correct = test_code(formatted_code)

        # Create json formatting for patch
        json_patches.append(
            {
                "code": formatted_code,
                "is_correct": is_code_correct,
                "conversation": conv,
            }
        )

        # Calculate tokens generated
        tokens_generated += len(
            tokenizer.encode(formatted_code, add_special_tokens=False)
        )

    # Calculate tokens per second
    tokens_per_second = tokens_generated / time_to_complete

    # Format json response
    ret = {
        "patches": json_patches,
        "tokens_generated": tokens_generated,
        "tokens_per_second": tokens_per_second,
        "time_to_complete": time_to_complete,
    }

    return ret


def load_model(
    model_cache_dir: str, chat_template: str, model_id: str
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    print("Model Loading starting")

    # Load model and tokenizer on GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda:0",
        trust_remote_code=True,
        cache_dir=model_cache_dir,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, device_map="cuda:0", cache_dir=model_cache_dir
    )

    # Load chat template
    tokenizer.chat_template = chat_template

    print("Model Loading complete\n")

    return model, tokenizer


def evaluate(
    tokenizer: PreTrainedTokenizer, model: PreTrainedModel, patches_per_bug
) -> List[dict]:
    prompts = get_prompts()
    bug_json = {}

    for bug_nr, prompt in enumerate(prompts):
        start_time = time.time()

        llm_response, tokenized_chat = batch_completion(
            model, tokenizer, prompt, patches_per_bug, bug_nr, len(prompts)
        )

        end_time = time.time()
        elapsed_time = end_time - start_time

        formatted_response = format_response(
            elapsed_time,
            llm_response,
            tokenizer,
            tokenizer.decode(tokenized_chat),
            prompt,
        )

        key = "bug nr: " + str(bug_nr + 1)
        bug_json[key] = formatted_response

    print("\n")

    return bug_json


@torch.inference_mode()
def batch_completion(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt,
    patch_size,
    bug_nr,
    total_bugs,
) -> Tuple[List[str], torch.Tensor]:
    input = None
    batch_completions = []
    gen_cfg = GenerationConfig.from_model_config(model.config)
    
    for patch_nr in range(1, patch_size+1):
        input = tokenizer.apply_chat_template(prompt, return_tensors="pt").to("cuda")

        generated_ids = model.generate(
            input,
            use_cache=True,
            max_new_tokens=args.max_new_tokens,
            generation_config=gen_cfg,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        batch_completions.append(tokenizer.decode(generated_ids[0]))

        # Print progress
        print_progress_bar(
            (bug_nr * patch_size) + patch_nr, total_bugs * patch_size
        )

    return batch_completions, input[0]


def print_progress_bar(
    bug_iteration: int,
    bug_total: int,
    prefix: str = "Progress",
    suffix: str = "Complete",
    length: int = 50,
    fill: str = "█",
) -> None:
    bug_percent = ("{0:.1f}").format(100 * (bug_iteration / float(bug_total)))
    bug_filled_length = int(length * bug_iteration // bug_total)
    bug_bar = fill * bug_filled_length + "-" * (length - bug_filled_length)

    print(
        f"\r{prefix} |Bug {bug_bar}|{bug_iteration}/{bug_total} |{bug_percent}% {suffix}",
        end="",
        flush=True,
    )


if __name__ == "__main__":
    # Initialize arguments
    args = parse_args()

    # Get the selected template set
    chat_template = get_chat_template(args.template_set)

    # Get model and prompt
    model, tokenizer = load_model(args.model_path, chat_template, args.model_id)

    # Generate answers
    json_data = evaluate(tokenizer, model, args.patches_per_bug)

    # Write JSON into a file
    json_to_file(json_data, args.json_path)
