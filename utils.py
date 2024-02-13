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
    """
    Prints a progress bar to track the progress of a process.

    :param current_iteration: The current iteration number.
    :param total_iterations: The total number of iterations.
    :param prefix: Optional string prefix to display before the progress bar.
    :param suffix: Optional string suffix to display after the progress bar.
    :param fill: Character used to fill the progress bar.
    """
    terminal_columns = shutil.get_terminal_size().columns  # Requires `import shutil`

    percentage = 1.0
    if total_iterations != 0:
        percentage = current_iteration / float(total_iterations)
    # Calculate the available space for the progress bar
    available_space = terminal_columns - len(
        f"\r{prefix} ||{current_iteration}/{total_iterations} ||100.0% {suffix}"
    )

    # Calculate the filled length of the progress bar
    filled_length = int(available_space * (percentage))

    # Build the progress bar
    progress_bar = fill * filled_length + "-" * (available_space - filled_length)

    # Format the percentage complete
    percent_complete = ("{0:.1f}").format(100 * (percentage))

    # Print the progress bar
    print(
        f"\r{prefix} |{progress_bar}|{current_iteration}/{total_iterations} |{percent_complete}% {suffix}",
        end="",
        flush=True,
    )


def save_json(data: dict, json_path: str) -> None:
    """
    Writes JSON data to a file.

    :param data: The JSON data to be written.
    :param json_path: The path to the JSON file.
    """
    # Ensure the directory structure exists
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # Write the dictionary to the JSON file
    with open(json_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def get_pass_k(total_list, correct_list, patches_per_bug):
    """
    Calculates the pass@k metric.

    :param total_list: List of total attempts made for each patch.
    :param correct_list: List of correct attempts made for each patch.
    :param patches_per_bug: Number of patches per bug.
    :return: Tuple containing the pass@1 and pass@k values.
    """
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
