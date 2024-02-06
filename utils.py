import json
import shutil
import os
import json
import numpy as np
from human_eval.evaluation import estimate_pass_at_k


def print_progress_bar(
    current_iteration: int,
    total_iterations: int,
    prefix: str = "Progress",
    suffix: str = "Complete",
    fill: str = "█",
) -> None:
    """Prints a progress bar to track the progress of a process."""
    terminal_columns = shutil.get_terminal_size().columns  # Requires `import shutil`

    # Calculate the available space for the progress bar
    available_space = terminal_columns - len(
        f"\r{prefix} ||{current_iteration}/{total_iterations} ||100.0% {suffix}"
    )

    # Calculate the filled length of the progress bar
    filled_length = int(available_space * (current_iteration / total_iterations))

    # Build the progress bar
    progress_bar = fill * filled_length + "-" * (available_space - filled_length)

    # Format the percentage complete
    percent_complete = ("{0:.1f}").format(
        100 * (current_iteration / float(total_iterations))
    )

    # Print the progress bar
    print(
        f"\r{prefix} |{progress_bar}|{current_iteration}/{total_iterations} |{percent_complete}% {suffix}",
        end="",
        flush=True,
    )


def write_dict_to_json(data: dict, json_path: str) -> None:
    """Writes a dictionary to a JSON file with indentation."""
    # Ensure the directory structure exists
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # Write the dictionary to the JSON file
    with open(json_path, "w") as json_file:
        json.dump(data, json_file, indent=6)


def get_summary(model_name: str, model_result: str, configurator):
    model_passed = 0
    model_failed = 0
    model_tokens_generated = 0
    model_tokens_sec = 0
    model_time = 0
    model_plausable_patch = 0
    model_total_patches = 0
    model_syntax_error = 0
    model_correct_list = []
    model_total_list = []
    model_total_list
    dataset_list = {}

    for model_item in model_result:
        dataset_name, dataset_item = model_item.popitem()
        dataset_passed = 0
        dataset_failed = 0
        dataset_tokens_generated = 0
        dataset_tokens_sec = 0
        dataset_time = 0
        dataset_plausable_patch = 0
        dataset_total_patches = 0
        dataset_syntax_error = 0
        dataset_correct_list = []
        dataset_total_list = []

        if list(dataset_item[0].keys())[0] == "Avg Pass@1":
            dataset_item.pop(0)
        for item_dict in dataset_item:
            item = list(item_dict.values())[0]
            dataset_time += item["time_s"]
            dataset_tokens_generated += item["tokens_generated"]
            dataset_tokens_sec += item["tokens/s"]
            bug_plausable_patch = 0

            for patch_dict in item["patches"]:
                dataset_total_patches += 1
                passed = patch_dict["test_result"]["Passed"]
                failed = patch_dict["test_result"]["Failed"]
                if patch_dict["test_result"]["syntax_error"]:
                    dataset_syntax_error += 1
                if isinstance(passed, int) and isinstance(failed, int):
                    dataset_passed += passed
                    dataset_failed += failed

                    if passed > 0 and failed == 0:
                        bug_plausable_patch += 1

            dataset_correct_list.append(bug_plausable_patch)
            dataset_total_list.append(len(item["patches"]))
            dataset_plausable_patch += bug_plausable_patch

        model_passed += dataset_passed
        model_failed += dataset_failed
        model_plausable_patch += dataset_plausable_patch
        model_total_patches += dataset_total_patches
        model_correct_list += dataset_correct_list
        model_total_list += dataset_total_list
        model_tokens_generated += dataset_tokens_generated
        model_tokens_sec += dataset_tokens_sec
        model_time += dataset_time
        model_syntax_error += dataset_syntax_error

        pass_1, pass_k = get_pass_k(
            dataset_total_list, dataset_correct_list, configurator.patches_per_bug
        )
        dataset_list[dataset_name] = {
            "Passed tests": dataset_passed,
            "Failed tests": dataset_failed,
            "Plausable patches": dataset_plausable_patch,
            "Syntax errors": dataset_syntax_error,
            "Total patches": dataset_total_patches,
            "Tokens generated": dataset_tokens_generated,
            "Tokens per patch": dataset_tokens_generated / dataset_total_patches,
            "Tokens/sec": dataset_tokens_sec,
            "Time taken(sec)": dataset_time,
            f"new pass@1": pass_1,
            f"new pass@{configurator.patches_per_bug}": pass_k,
        }
    model_summary = {}
    pass_1, pass_k = get_pass_k(
        model_total_list, model_correct_list, configurator.patches_per_bug
    )

    model_summary[model_name] = {
        "Passed tests": model_passed,
        "Failed tests": model_failed,
        "Plausable patches": model_plausable_patch,
        "Total patches": model_total_patches,
        "Syntax errors": model_syntax_error,
        "Tokens generated": model_tokens_generated,
        "Tokens per patch": model_tokens_generated / model_total_patches,
        "Tokens/sec": model_tokens_sec,
        "Time taken(sec)": model_time,
        f"new pass@1": pass_1,
        f"new pass@{configurator.patches_per_bug}": pass_k,
        "Datasets summary": dataset_list,
        "Parameters summary": {
            "Max length": configurator.max_length,
            "Temperature": configurator.temperature,
            "Top p": configurator.top_p,
            "Patches per bug": configurator.patches_per_bug,
        },
    }
    return model_summary


def get_pass_k(total_list, correct_list, patches_per_bug):
    pass_1 = estimate_pass_at_k(np.array(total_list), np.array(correct_list), 1)
    pass_1 = pass_1.tolist()
    pass_1 = sum(pass_1) / len(pass_1)
    pass_k = estimate_pass_at_k(
        np.array(total_list),
        np.array(correct_list),
        patches_per_bug,
    )
    pass_k = pass_k.tolist()
    pass_k = sum(pass_k) / len(pass_k)
    return pass_1, pass_k
