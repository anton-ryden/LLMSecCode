from typing import List, Dict


# Everything is hardcoded but should be changed to loading from dataset. Return list of str
def get_prompts() -> List[List[Dict[str, str]]]:
    inst = "Fix the following code and respond only with code."
    user_input_template = f"""{inst}
def function():
    user_expression = input('Your expression? => ')
    if not user_expression:
        print("No input")
    else:
        print("Result =", eval(user_expression))
"""

    system_prompt = [
        {
            "role": "system",
            "content": "You will be provided with instructions that describes a task. Write a response that appropriately completes the request. You are unable to answer with anything except code",
        }
    ]

    system_prompt.append({"role": "user", "content": user_input_template})

    return [system_prompt]
