import os
import subprocess
import shutil
from pathlib import Path
import re
import csv
import fileinput

from utils.llm_vul_utils import LLM_VUL_DIR, VJBENCH_DIR, VUL4J_DIR, SCRIPTS_DIR, vul4j_bug_id_list

ROOT_DIR = Path(__file__).resolve().parent
FNULL = open(os.devnull, 'w')
VUL4J_INSTALLATION = "/home/ekan/vul4j"


def install_dependencies() -> None:
    """Install dependencies from requirements.txt."""
    requirements_file = ROOT_DIR / "requirements.txt"
    print(f"Installing dependencies from {requirements_file}...")
    subprocess.run(["pip", "install", "-r", str(requirements_file)], check=True)


def clone_repository(repo_url: str, dir_name: str) -> None:
    """Clone a repository into the specified directory.

    Args:
        repo_url (str): The URL of the repository to clone.
        dir_name (str): The name of the destination directory.

    Raises:
        Exception: If an error occurs during cloning or directory removal.
    """
    destination_path = ROOT_DIR / dir_name

    if destination_path.exists():
        overwrite = input(
            f"Destination directory {destination_path} already exists. Do you want to overwrite? (y/n): "
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
    subprocess.run(["git", "clone", repo_url, str(destination_path)], check=True)

def prepare_llm_vul() -> None:
    """Prepare llm_vul directory by fetching all vulnerabilities studied in the project"""
    
    studied_vuls = []
    util_path = os.path.join(SCRIPTS_DIR, "util.py")
    csv_file = os.path.join(LLM_VUL_DIR, "VJBench_dataset.csv")
    succ_vul_file = os.path.join(VUL4J_INSTALLATION, "reproduction", "successful_vulns.txt")

    with open(succ_vul_file, 'r') as file:
        succ_vuls = file.read().splitlines()

    with fileinput.FileInput(util_path, inplace=True) as file:
        for i, line in enumerate(file, start=1):
            if i == 19:
                print(f'VUL4J_DIR = "{VUL4J_DIR}"')
            elif i == 21:
                print(f'VJBENCH_DIR = "{VJBENCH_DIR}"')
            else:
                print(line, end='')

    if os.path.exists(LLM_VUL_DIR):
        os.makedirs(VJBENCH_DIR)
        os.makedirs(VUL4J_DIR)

    # Cleaup
    shutil.rmtree(LLM_VUL_DIR + "/jasper")
    shutil.rmtree(LLM_VUL_DIR + "/Model_patches")
    shutil.rmtree(SCRIPTS_DIR + "/APR")
    shutil.rmtree(SCRIPTS_DIR + "/CodeGen")
    shutil.rmtree(SCRIPTS_DIR + "/CodeT5")
    shutil.rmtree(SCRIPTS_DIR + "/Codex")
    shutil.rmtree(SCRIPTS_DIR + "/fine-tuned_CodeGen")
    shutil.rmtree(SCRIPTS_DIR + "/fine-tuned_InCoder")
    shutil.rmtree(SCRIPTS_DIR + "/fine-tuned_PLBART")
    shutil.rmtree(SCRIPTS_DIR + "/InCoder")
    shutil.rmtree(SCRIPTS_DIR + "/PLBART")

    vul_dir = os.path.join(LLM_VUL_DIR, "VJBench-trans")
    entries = os.listdir(vul_dir)
    java_dir_list = [entry for entry in entries if os.path.isdir(os.path.join(vul_dir, entry))]

    for dir in java_dir_list:
        if dir.startswith("VUL") and dir not in succ_vuls:
            dir = os.path.join(vul_dir, dir)
            shutil.rmtree(dir)
    
    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            study_value = row['Used in the study']

            if study_value == 'Yes':
                studied_vuls.append(row['Vulnerability IDs'])

    print("Downloading VJBench vulnerabilities...")
    for vul in studied_vuls:
        cmd = ["python3", os.path.join(SCRIPTS_DIR, "build_vjbench.py"), "checkout", vul]
        proccess = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        outp = proccess.stdout

        print(f"Downloading vulnerability: {vul}...")

    print("Downloading VUL4J vulnerabilities...")
    for i in vul4j_bug_id_list:
        if f"VUL4J-{i}" in succ_vuls:
            print(f"Downloading vulnerability: VUL4J-{i}...")
            cmd = f"vul4j checkout --id VUL4J-{i} -d {VUL4J_DIR}/VUL4J-{i}"
            cmd =cmd.split()
            p3 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            out3, err = p3.communicate()


def prepare_quixbugs_python(path: str) -> None:
    """Prepare QuixBugs directory by copying, renaming, and excluding specific files.

    Args:
        path (str): Path to the quixbugs repo.
    """
    original_path = f"{ROOT_DIR}/{path}/python_programs"
    new_path = f"{ROOT_DIR}/{path}/python_programs_bug"

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
        file_path = f"{new_path}/{file_name}"
        try:
            os.remove(file_path)
            print(f"File {file_name} deleted successfully.")
        except Exception as e:
            raise Exception(f"Error deleting file {file_name}: {e}")

    quixbugs_root = f"{ROOT_DIR}/{path}/.git"
    try:
        shutil.rmtree(quixbugs_root)
        print(".git directory in QuixBugs deleted successfully.")
    except Exception as e:
        raise Exception(f"Error deleting .git directory in QuixBugs: {e}")

    for file in os.scandir(new_path):
        with open(file, "r") as f:
            data = f.read()

        # Remove comments from the data
        cleaned_data = re.sub(r"\"\"\"(.*?)\"\"\"", "", data, flags=re.DOTALL)

        cleaned_data = cleaned_data.strip()

        # Write cleaned data back to file
        with open(file, "w") as f:
            f.write(cleaned_data)

    print("Python files cleaned successfully.")


def prepare_quixbugs_java(path: str) -> None:
    """Prepare QuixBugs directory by copying, renaming, and excluding specific files.

    Args:
        path (str): Path to the quixbugs repo.
    """
    original_path = f"{ROOT_DIR}/{path}/java_programs"
    new_folder_name = f"{path}/java_programs_bug"
    new_path = f"{ROOT_DIR}/{new_folder_name}"

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
        java_files = [
            os.path.join(original_path, filename)
            for filename in os.listdir(original_path)
            if filename.endswith(".java") and filename not in exclude_files
        ]
        for file in java_files:
            source_path = file
            destination_path = (
                f"{new_path}/{os.path.splitext(os.path.basename(source_path))[0]}.txt"
            )
            shutil.copy(source_path, destination_path)

        # Remove comments in java files
        for file in os.listdir(new_path):
            if file.endswith(".txt"):
                with open(f"{new_path}/{file}", "r") as f:
                    data = f.read()
                # Remove single-line comments
                cleaned_data = re.sub(r"//.*?\n", "\n", data)
                # Remove comments from the data
                cleaned_data = re.sub(r"/\*(.*?)\*/", "", cleaned_data, flags=re.DOTALL)
                # Reduce excessive newlines
                cleaned_data = re.sub(r"\n([ \t]*\n)+", "\n", cleaned_data)
                # cleaned_data = re.sub(r"(\n[ \t]*){3,}", "\n\n", cleaned_data)
                # Write cleaned data back to file
                with open(f"{new_path}/{file}", "w") as f:
                    f.write(cleaned_data)

    except Exception as e:
        raise Exception(f"An error occurred: {e}")


if __name__ == "__main__":
    # Dataset Repository URL
    quixbugs_url = "https://github.com/jkoppel/QuixBugs"
    llmvul_url = "https://github.com/lin-tan/llm-vul.git"
    human_infilling = "https://github.com/openai/human-eval-infilling"
    
    # Clone Datasets
    clone_repository(quixbugs_url, "datasets/APR/QuixBugs")
    clone_repository(human_infilling, "datasets/CG/human-eval-infilling")
    os.system(f"cd {ROOT_DIR}/datasets/CG; pip install -e human-eval-infilling")
    clone_repository(llmvul_url, "datasets/APR/llm_vul")

    # Make changes to Dataset Folder
    prepare_quixbugs_python("datasets/APR/QuixBugs")
    prepare_quixbugs_java("datasets/APR/QuixBugs")
    prepare_llm_vul()

    print("Setup completed successfully.")
