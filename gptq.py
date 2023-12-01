#pip install -q -U transformers peft accelerate optimum
#pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu117/
from config import load_config

import os
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load config so it is accessible in file
cfg = load_config()

def load_model(cache_dir):
    torch.cuda.empty_cache()

    model_id = cfg.get('model', {}).get('huggingface_id') # Get model name from config file

    # Load model and tokenizer on gpu
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0", trust_remote_code=False, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, device_map="cuda:0", cache_dir=cache_dir)

    # Load chat template
    #tokenizer.chat_template = cfg.get('chat', {}).get('template')

    return model, tokenizer

def get_prompts():
    temp0 = """user_input = input("Enter your name: ")\ngreeting = "Hello, " + user_input + "!"\nprint(greeting)"""

    temp1="""user_input = input("Enter your password: ")\nprint(f"Password: {user_input}")"""

    system_prompt0 = [{
            "role": "system",
            "content": "You are an automatic program repair bot that improves code. Always answer with code and only code.",
    }]
    system_prompt1 = [{
        "role": "system",
        "content": "You are an automatic program repair bot that improves code. Always answer with code and only code.",
    }]
    
    system_prompt0.append({"role": "user", "content": temp0})
    system_prompt1.append({"role": "user", "content": temp1})

    prompt = [system_prompt0, system_prompt1]

    return  prompt

def remove_instruction_prompt(answer_text):
    # Find the index of [/INST]
    inst_end_index = answer_text.find("[/INST]")

    # Remove everything before and including [/INST]
    if inst_end_index != -1:
        cleaned_answer = answer_text[inst_end_index + len("[/INST]"):].strip()
        return cleaned_answer
    else:
        return answer_text.strip()
    
def extract_code(input_string):
    start_index = input_string.find("```")
    end_index = input_string.rfind("```")

    if start_index != -1 and end_index != -1 and start_index < end_index:
        return input_string[start_index + 3:end_index].strip()
    else:
        return None

def generate_answers(tokenizer, model):
    prompts = get_prompts() # Get prompts

    answers = []

    for i, prompt in enumerate(prompts):
        start_time = time.time()

        tokenized_chat = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt").to('cuda')

        outputs = model.generate(tokenized_chat, max_new_tokens=300)
        answer_text = tokenizer.decode(outputs[0])

        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Token calculation for the prompt without instruction
        answer_without_instruction = remove_instruction_prompt(answer_text)
        tokens_generated_prompt = len(tokenizer.encode(answer_without_instruction))
        tokens_per_second_prompt = tokens_generated_prompt / elapsed_time
        json_without_prompt = [{"answer" : answer_without_instruction,
                            "tokens_generated" : tokens_generated_prompt, 
                            "tokens_per_second" : tokens_per_second_prompt}]

        # Token calculation for fixed code
        # Extract fixed code by finding content within triple quotes
        fixed_code = extract_code(answer_without_instruction)
        tokens_generated_fixed = 0
        if fixed_code != None:
            tokens_generated_fixed = len(tokenizer.encode(fixed_code))

        tokens_per_second_fixed_code = tokens_generated_fixed / elapsed_time
        json_code = [{"answer" : fixed_code,
                        "tokens_generated" : tokens_generated_fixed, 
                        "tokens_per_second" : tokens_per_second_fixed_code}]

        prompt.append({"role": "assistant", "content": answer_without_instruction})
        answers.append({f"question nr: {i}" : {
            "conversation_json": prompt,
            "without_instruction" : json_without_prompt,
            "fixed_code": json_code,
        }})

    return answers

if __name__ == "__main__":
    # Change chache to the model directory
    dirname = os.path.dirname(__file__) # This files path
    cache_dir = os.path.join(dirname, 'models')

    # Get model and prompt
    model, tokenizer = load_model(cache_dir)

    # Generate answers
    json_data = generate_answers(tokenizer, model)

    print(json.dumps(json_data, indent=4))
    
    json_path = os.path.join(dirname, 'answers.json')
    with open(json_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)
