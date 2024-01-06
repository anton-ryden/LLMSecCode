import json
import shutil
import os

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
    available_space = terminal_columns - len(f"\r{prefix} ||{current_iteration}/{total_iterations} ||100.0% {suffix}")

    # Calculate the filled length of the progress bar
    filled_length = int(available_space * (current_iteration / total_iterations))

    # Build the progress bar
    progress_bar = fill * filled_length + "-" * (available_space - filled_length)

    # Format the percentage complete
    percent_complete = ("{0:.1f}").format(100 * (current_iteration / float(total_iterations)))

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
