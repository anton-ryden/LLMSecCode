import os
import shutil
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def install_dependencies() -> None:
    """Install dependencies from requirements.txt."""
    requirements_file = os.path.join(SCRIPT_DIR, "requirements.txt")
    print(f"Installing dependencies from {requirements_file}...")
    os.system(f"pip install -r {requirements_file}")


def clone_repository(repo_url: str, dir_name: str) -> None:
    """Clone a repository into the specified directory.

    Args:
        repo_url (str): The URL of the repository to clone.
        dir_name (str): The name of the destination directory.

    Raises:
        Exception: If an error occurs during cloning or directory removal.
    """
    destination_path = os.path.join(SCRIPT_DIR, dir_name)

    if os.path.exists(destination_path):
        overwrite = input(
            "Destination directory already exists. Do you want to overwrite? (y/n): "
        ).lower()
        if overwrite != "y":
            print("Skipping cloning.")
            return
        else:
            print(f"Overwriting existing directory at {destination_path}...")
            try:
                shutil.rmtree(destination_path)
            except Exception as e:
                raise Exception(f"Error removing existing directory: {e}")

    print(f"Cloning repository from {repo_url} to {destination_path}...")
    os.system(f"git clone {repo_url} {destination_path}")


def prepare_quixbugs_python() -> None:
    """Prepare QuixBugs directory by copying, renaming, and excluding specific files."""
    original_path = os.path.join(SCRIPT_DIR, "QuixBugs/python_programs")
    new_folder_name = "QuixBugs/python_programs_bug"
    new_path = os.path.join(SCRIPT_DIR, new_folder_name)

    print(f"Copying and renaming directory from {original_path} to {new_path}...")
    shutil.copytree(original_path, new_path)
    files_to_exclude = [
        "node.py",
        "breadth_first_search_test.py",
        "depth_first_search_test.py",
        "detect_cycle_test.py",
        "minimum_spanning_tree_test.py",
        "reverse_linked_list_test.py",
        "shortest_path_lengths_test.py",
        "shortest_path_length_test.py",
        "shortest_paths_test.py",
        "topological_ordering_test.py",
    ]

    for file_name in files_to_exclude:
        file_path = os.path.join(new_path, file_name)
        try:
            os.remove(file_path)
            print(f"File {file_name} deleted successfully.")
        except Exception as e:
            raise Exception(f"Error deleting file {file_name}: {e}")

    quixbugs_root = os.path.join(SCRIPT_DIR, "QuixBugs/.git")
    try:
        shutil.rmtree(quixbugs_root)
        print(".git directory in QuixBugs deleted successfully.")
    except Exception as e:
        raise Exception(f"Error deleting .git directory in QuixBugs: {e}")

    # Remove comments in python files
    for file in os.listdir(new_path):
        with open(os.path.join(new_path, file), "r") as f:
            data = f.read()
            # Remove comments from the data
            cleaned_data = re.sub(r"\"\"\"(.*?)\"\"\"", "", data, flags=re.DOTALL)
            # Remove excess newlines
            cleaned_data = re.sub(r"\n{2,}", "\n", cleaned_data)
            # Write cleaned data back to file
            with open(os.path.join(new_path, file), "w") as f:
                f.write(cleaned_data)


def prepare_quixbugs_java() -> None:
    """Prepare QuixBugs directory by copying, renaming, and excluding specific files."""
    original_path = os.path.join(SCRIPT_DIR, "QuixBugs/java_programs")
    new_folder_name = "QuixBugs/java_programs_bug"
    new_path = os.path.join(SCRIPT_DIR, new_folder_name)

    exclude_files = [
        "Node.java",
        "WeightedEdge.java",
        "Node.class",
        "WeightedEdge.class",
    ]

    try:
        # Create the destination directory if it doesn't exist
        os.makedirs(new_path, exist_ok=True)

        # Move .java files to the destination directory
        for filename in os.listdir(original_path):
            source_path = os.path.join(original_path, filename)
            destination_path = os.path.join(new_path, filename.split(".")[0] + ".txt")

            # Check if the file is a .java file and not in the exclude list
            if filename.endswith(".java") and filename not in exclude_files:
                # if filename.endswith(".java") and filename:
                shutil.copy(source_path, destination_path)

        # Remove comments in java files
        for file in os.listdir(new_path):
            with open(os.path.join(new_path, file), "r") as f:
                data = f.read()
                # Remove comments from the data
                cleaned_data = re.sub(r"/\*(.*?)\*/", "", data, flags=re.DOTALL)
                # Remove excess newlines
                cleaned_data = re.sub(r"\n{2,}", "\n", cleaned_data)
                # Write cleaned data back to file
                with open(os.path.join(new_path, file), "w") as f:
                    f.write(cleaned_data)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # QuixBugs Repository URL
    quixbugs_url = "https://github.com/jkoppel/QuixBugs"

    # Clone QuixBugs
    clone_repository(quixbugs_url, "QuixBugs")

    # Make changes to QuixBugs Folder
    prepare_quixbugs_python()
    prepare_quixbugs_java()
    print("Setup completed successfully.")
