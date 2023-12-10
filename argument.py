import argparse


# Note codewizard does not work atm
def parse_args():
    parser = argparse.ArgumentParser(description="This is option of arguments.")
    parser.add_argument(
        "--template_set",
        choices=["llama", "wizardcode"],
        default="llama",
        help="Choose the set of templates to use (llama or codewizard).\n Default is %(default)s.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="TheBloke/CodeLlama-7B-Instruct-GPTQ",
        help="Specify the Hugging Face model ID ex: TheBloke/CodeLlama-7B-Instruct-GPTQ.\n Default is %(default)s.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models",
        help="Specify where to look and save models to.\n Default is %(default)s.",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="./answers.json",
        help="Specify path and name for the JSON file. Default is %(default)s.",
    )
    parser.add_argument(
        "--patches_per_bug",
        type=int,
        default=2,
        help="The number of patches to generate per bug",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=500,
        help="The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.\n Default is %(default)s.",
    )
    return parser.parse_args()


def get_chat_template(template_set):
    template_llama = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif true == true and not '<<SYS>>' in messages[0]['content'] %}{% set loop_messages = messages %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ ' ' + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
    template_codewizard = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{{ messages[0]['content'].strip() }}{% elif message['role'] == 'user' %}{{ bos_token + message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ ' '  + message['content'].strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
    
    if template_set == "llama":
        return template_llama
    elif template_set == "wizardcode":
        return template_codewizard
    else:
        raise ValueError(f"Invalid template set: {template_set}")
