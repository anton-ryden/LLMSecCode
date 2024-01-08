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

def prepare_quixbugs_java() -> None:
    """Prepare QuixBugs directory by copying, renaming, and excluding specific files."""
    original_path = os.path.join(SCRIPT_DIR, "QuixBugs/java_programs")
    new_folder_name = "java_programs_bug"
    new_path = os.path.join(SCRIPT_DIR, new_folder_name)
    exclude_files = ["Node.java", "WeightedEdge.java", "Node.class", "WeightedEdge.class"]
    original_path_test = os.path.join(SCRIPT_DIR, "QuixBugs/java_testcases/junit/")
    new_test_folder_name = "java_testcases/junit/"
    new_path_test = os.path.join(SCRIPT_DIR, new_test_folder_name)
    additional_test_remove = os.path.join(SCRIPT_DIR, 'QuixBugs/java_testcases')
    try:
        for filename in os.listdir(additional_test_remove):
            if filename.endswith(".java"):
                file_path = os.path.join(additional_test_remove, filename)
                os.remove(file_path)
        # Create the destination directory if it doesn't exist
        os.makedirs(new_path, exist_ok=True)

        # Move .java files to the destination directory
        for filename in os.listdir(original_path):
            source_path = os.path.join(original_path, filename)
            destination_path = os.path.join(new_path, filename)

            # Check if the file is a .java file and not in the exclude list
            if filename.endswith(".java") and filename not in exclude_files:
                shutil.move(source_path, destination_path)

        # Remove .class files from the source directory
        for filename in os.listdir(original_path):
            file_path = os.path.join(original_path, filename)

            # Check if the file is a .class file and not in the exclude list
            if filename.endswith(".class") and filename not in exclude_files:
                os.remove(file_path)
        
        # Create the destination directory if it doesn't exist
        os.makedirs(new_path_test, exist_ok=True)

        # Move all files and folders from source to destination
        for item in os.listdir(original_path_test):
            if item.endswith("TEST.java"):
                source_path = os.path.join(original_path_test, item)
                destination_path = os.path.join(new_path_test, item)
                shutil.move(source_path, destination_path)


        # Remove comments in java files
        for file in os.listdir(new_path):
            with open(os.path.join(new_path, file), "r") as f:
                data = f.read()
                # Remove comments from the data
                cleaned_data = re.sub(r'/\*(.*?)\*/', '', data, flags=re.DOTALL)
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
