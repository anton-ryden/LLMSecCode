import json
import time
import copy
from transformers.generation import GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from argument import parse_args, get_chat_template
from get_prompts import *
from test_code import *


def json_to_file(data, json_path):
    # Write JSON into a file
    with open(json_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def extract_code(input_string):
    start_index = input_string.find("```")
    end_index = input_string.rfind(
        "```"
    )  # Start searching after the first triple backticks

    if start_index != -1 and end_index != -1:
        input_string = input_string[start_index + 3 : end_index]
        input_string = input_string.lstrip()
        return input_string
    else:
        return None


def format_response(elapsed_time, answer_text, patch_nr, prompt, tokenized_chat):
    # Remove instructions from response
    answ_no_ins = answer_text.replace(tokenized_chat, "")
    answ_no_ins = answ_no_ins[: len(answ_no_ins) - len(tokenizer.eos_token)]
    # Format response and calculate token/s
    tokens_generated_prompt = len(tokenizer.encode(answ_no_ins))
    tokens_per_second_prompt = tokens_generated_prompt / elapsed_time
    json_without_prompt = [
        {
            "answer": answ_no_ins,
            "tokens_generated": tokens_generated_prompt,
            "tokens_per_second": tokens_per_second_prompt,
        }
    ]

    # Format response and calculate token/s
    fixed_code = extract_code(answ_no_ins)

    # If response is formatted incorrectly
    tokens_generated_fixed = 0
    if fixed_code is not None:
        tokens_generated_fixed = len(tokenizer.encode(fixed_code))

    tokens_per_second_fixed_code = tokens_generated_fixed / elapsed_time
    json_code = [
        {
            "patch": fixed_code,
            "tokens_generated": tokens_generated_fixed,
            "tokens_per_second": tokens_per_second_fixed_code,
        }
    ]

    prompt.append({"role": "assistant", "content": answ_no_ins})

    is_code_fixed = test_code(fixed_code)

    ret = {
        f"patch nr: {patch_nr}": {
            "conversation_json": prompt,
            "without_instruction": json_without_prompt,
            "fixed_code": json_code,
            "time_s": elapsed_time,
            "fixed error": is_code_fixed,
        }
    }

    return ret


def load_model(model_cache_dir, chat_template, model_id):
    print("Model Loading starting")

    # Load model and tokenizer on GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda:0",
        trust_remote_code=True,
        cache_dir=model_cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, device_map="cuda:0", cache_dir=model_cache_dir
    )
    # Load chat template
    tokenizer.chat_template = chat_template

    print("Model Loading complete\n")

    return model, tokenizer


def generate_answers(tokenizer, model):
    gen_cfg = GenerationConfig.from_model_config(model.config)

    prompts = get_prompts()  # Get prompts
    answers = []

    tokenized_chats = [
        tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=False, return_tensors="pt"
        ).to("cuda")
        for prompt in prompts
    ]

    for bug_nr, prompt in enumerate(prompts):
        patches = []

        for patch_nr in range(1, args.patches_per_bug + 1):
            start_time = time.time()

            tokenized_chat = tokenized_chats[bug_nr]

            outputs = model.generate(
                tokenized_chat,
                max_new_tokens=args.max_new_tokens,
                generation_config=gen_cfg,
            )
            llm_response = tokenizer.decode(
                outputs[0], clean_up_tokenization_spaces=False
            )

            end_time = time.time()
            elapsed_time = end_time - start_time

            formatted_response = format_response(
                elapsed_time,
                llm_response,
                patch_nr,
                copy.copy(prompt),
                tokenizer.decode(tokenized_chat[0]),
            )
            patches.append(formatted_response)

            # Print progress
            print_progress_bar(bug_nr, len(prompts), patch_nr, args.patches_per_bug)

        print_progress_bar(bug_nr + 1, len(prompts), patch_nr, args.patches_per_bug)
        answers.append({f"bug nr: {bug_nr +1}": patches})

    print("\n")

    return answers


def print_progress_bar(
    bug_iteration,
    bug_total,
    patch_iteration,
    patch_total,
    prefix="Progress",
    suffix="Complete",
    length=50,
    fill="█",
):
    bug_percent = ("{0:.1f}").format(100 * (bug_iteration / float(bug_total)))
    patch_percent = ("{0:.1f}").format(100 * (patch_iteration / float(patch_total)))

    bug_filled_length = int(length * bug_iteration // bug_total)
    patch_filled_length = int(length * patch_iteration // patch_total)

    bug_bar = fill * bug_filled_length + "-" * (length - bug_filled_length)
    patch_bar = fill * patch_filled_length + "-" * (length - patch_filled_length)

    print(
        f"\r{prefix} |Bug {bug_bar}| {bug_percent}% |Patch {patch_bar}| {patch_percent}% {suffix}",
        end="",
        flush=True,
    )


if __name__ == "__main__":
    # Initialize arguments
    args = parse_args()

    # Get the selected template set
    chat_template = get_chat_template(args.template_set)

    # Change cache to the model directory
    model_cache_dir = args.model_path

    # Get model and prompt
    model, tokenizer = load_model(model_cache_dir, chat_template, args.model_id)

    # Generate answers
    json_data = generate_answers(tokenizer, model)

    json.dumps(json_data, indent=4)

    # Write JSON into a file
    json_to_file(json_data, args.json_path)
