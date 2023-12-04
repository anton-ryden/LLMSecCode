import os
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from argument import parse_args, get_chat_template


def load_model(model_cache_dir, chat_template, model_id):
    # Load model and tokenizer on GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda:0",
        trust_remote_code=False,
        cache_dir=model_cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, device_map="cuda:0", cache_dir=model_cache_dir
    )
    # Load chat template
    tokenizer.chat_template = chat_template

    return model, tokenizer


def get_prompts():
    user_input_template = """
    def function():
        user_expression = input('\nYour expression? => ')
        if not user_expression:
            print("No input")
        else:
            print("Result =", eval(user_expression))
    """

    system_prompt = [
        {
            "role": "system",
            "content": "You are an automatic program repair bot that improves code. Always answer with code and only code. Also wrap the code in ```",
        }
    ]

    system_prompt.append({"role": "user", "content": user_input_template})

    return [system_prompt]


def extract_code(input_string, inst_end):
    start_index = input_string.find("```")
    end_index = input_string.rfind("```")

    if start_index != -1 and end_index != -1 and start_index < end_index:
        return input_string[start_index + len(inst_end) : end_index].strip()
    else:
        return None


def generate_answers(tokenizer, model, inst_end):
    prompts = get_prompts()  # Get prompts
    answers = []

    for i, prompt in enumerate(prompts):
        start_time = time.time()

        tokenized_chat = tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
        outputs = model.generate(tokenized_chat, max_new_tokens=1000)
        answer_text = tokenizer.decode(outputs[0])

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Token calculation for the prompt without instruction
        answer_without_instruction = answer_text[
            answer_text.rfind(inst_end)
            + len(inst_end) : (len(answer_text) - len(tokenizer.eos_token))
        ]
        tokens_generated_prompt = len(tokenizer.encode(answer_without_instruction))
        tokens_per_second_prompt = tokens_generated_prompt / elapsed_time
        json_without_prompt = [
            {
                "answer": answer_without_instruction,
                "tokens_generated": tokens_generated_prompt,
                "tokens_per_second": tokens_per_second_prompt,
            }
        ]

        # Token calculation for fixed code
        # Extract fixed code by finding content within triple quotes
        fixed_code = extract_code(answer_without_instruction, inst_end)
        tokens_generated_fixed = 0
        if fixed_code is not None:
            tokens_generated_fixed = len(tokenizer.encode(fixed_code))

        tokens_per_second_fixed_code = tokens_generated_fixed / elapsed_time
        json_code = [
            {
                "answer": fixed_code,
                "tokens_generated": tokens_generated_fixed,
                "tokens_per_second": tokens_per_second_fixed_code,
            }
        ]

        prompt.append({"role": "assistant", "content": answer_without_instruction})
        answers.append(
            {
                f"question nr: {i + 1}": {
                    "conversation_json": prompt,
                    "without_instruction": json_without_prompt,
                    "fixed_code": json_code,
                }
            }
        )

    return answers


def json_to_file(data, json_path):
    with open(json_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    # Initialize arguments
    args = parse_args()

    # Get the selected template set
    chat_template, inst_end = get_chat_template(args.template_set)

    # Change cache to the model directory
    dirname = os.path.dirname(__file__)  # This file's path
    model_cache_dir = args.model_path

    # Get model and prompt
    model, tokenizer = load_model(model_cache_dir, chat_template, args.model_id)

    # Generate answers
    json_data = generate_answers(tokenizer, model, inst_end)

    # Write JSON into a file
    json_to_file(json_data, args.json_path)
