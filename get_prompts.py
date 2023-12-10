# Everything is hardcoded but should be changed to loading from dataset. Return list of str
def get_prompts():
    user_input_template = """
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
            "content": "You are an automatic program repair bot. You provide code and wrap the code with ```",
        }
    ]

    system_prompt.append({"role": "user", "content": user_input_template})

    return [system_prompt]
