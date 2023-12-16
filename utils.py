import json


def print_progress_bar(
    current_iteration: int,
    total_iterations: int,
    prefix: str = "Progress",
    suffix: str = "Complete",
    length: int = 50,
    fill: str = "█",
) -> None:
    """Prints a progress bar to track the progress of a process."""
    percent_complete = ("{0:.1f}").format(100 * (current_iteration / float(total_iterations)))
    filled_length = int(length * current_iteration // total_iterations)
    progress_bar = fill * filled_length + "-" * (length - filled_length)

    print(
        f"\r{prefix} |{progress_bar}|{current_iteration}/{total_iterations} |{percent_complete}% {suffix}",
        end="",
        flush=True,
    )


def write_dict_to_json(data: dict, json_path: str) -> None:
    """Writes a dictionary to a JSON file with indentation."""
    with open(json_path, "w") as json_file:
        json.dump(data, json_file, indent=6)
