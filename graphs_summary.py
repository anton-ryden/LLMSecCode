import matplotlib.pyplot as plt
import json
import os
import numpy as np
import pandas as pd

def merge_json_files(input_folder, output_file):
    merged_data = {}

    # Get a list of all summary JSON files in the input folder
    prefixes = ["instruction", "infilling"]
    json_files = []

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if any(file.startswith(prefix + "_summary.json") for prefix in prefixes):
                json_files.append(os.path.join(root, file))
    json_files.sort()  # Sort the list of JSON files

    

    # Iterate over each JSON file and merge the data
    for json_file in json_files:
        with open(json_file) as file:
            data = json.load(file)
            filename = os.path.basename(json_file)
            filename = os.path.splitext(filename)[0]
            conversation_type = filename.split('_')[0]
            if conversation_type not in merged_data:
                merged_data[conversation_type] = []

            merged_data[conversation_type].append(data)

    output_directory = os.path.dirname(output_file)
    os.makedirs(output_directory, exist_ok=True)

    # Write the merged data to the output file
    with open(output_file, "w") as output:
        json.dump(merged_data, output, indent=6)
    

def plot_pass_at_k(result_directory, dataset):

    # Adjust fig size
    fig, ax = plt.subplots(figsize=(12, 6))

    all_pass_at_k_info = {}

    # Get the pass@k values of each model and config
    for json_file in os.listdir(result_directory):
        filename = os.path.basename(json_file)
        filename = os.path.splitext(filename)[0]
        model_name = filename.split('_')[0]
        if json_file.endswith(".json"):
            json_file_path = os.path.join(result_directory, json_file)
            with open(json_file_path) as file:
                data = json.load(file)

            pass_at_k_info = {}
            for config_type, config_data in data.items():
                for entry in config_data:
                    if entry["Name"] == dataset:
                        pass_at_k_info[config_type] = {}

                        for key, value in entry["Statistics"]["0"].items():
                            if key.startswith("Pass@"):
                                pass_at_k_info[config_type][key] = value

            all_pass_at_k_info[model_name] = {}
            all_pass_at_k_info[model_name] = pass_at_k_info

    # Plotting Pass@k
    df = pd.DataFrame.from_dict({(i, j): all_pass_at_k_info[i][j] 
                             for i in all_pass_at_k_info.keys() 
                             for j in all_pass_at_k_info[i].keys()},
                            orient='index')

# Plotting using Pandas plot
    fig, ax = plt.subplots()
    df.unstack().plot(kind='bar', ax=ax, width=0.8)

    ax.set_xlabel('Models and Configurations')
    ax.set_ylabel('Pass@k (%)')
    ax.set_title(f'Pass@k Values for {dataset}')
    ax.legend(title='Pass@k', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.savefig(f"./graphs/{dataset}_pass_at_k_plot.png", bbox_inches="tight")


if __name__ == "__main__":
    

    # Get the absolute path of the current file
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    default_result_path = os.path.join(current_file_path, "results/default/")
    merged_dir = os.path.join(default_result_path, "merged")

    for model_name in os.listdir(default_result_path):
        model_path = os.path.join(default_result_path, model_name)
        if os.path.isdir(model_path) and model_path != merged_dir:

            merge_json_files(
                model_path,
                default_result_path  + "/merged/" + model_name + "_merged.json",
            )
    
    # Define dataset (QuixBugs Python, QuixBugs Java, HumanEval)
    dataset = "QuixBugs Python"

    # Create directory for saving plotting results
    graph_directory = os.path.join(current_file_path, "graphs")
    os.makedirs(graph_directory, exist_ok=True)        
    
    # Plot graphs
    plot_pass_at_k(merged_dir, dataset)

    
