from typing import List
from configurator import Configurator
from dataset_loader.dataset_loader import DatasetLoader
from model_loader.model_loader import ModelLoader
from utils import write_dict_to_json

def evaluate(
    model_loaders: List[ModelLoader],
    dataset_loaders: List[DatasetLoader],
) -> List[dict]:
    # List to store the final evaluation results
    evaluation_result = []

    # Iterate over each model loader
    for model_loader in model_loaders:
        # List to store results for each dataset
        model_result = []

        # Iterate over each dataset loader
        for dataset_loader in dataset_loaders:
            # Generate model responses and timing
            responses, tot_time = model_loader.generate_answers(dataset_loader)

            # Extract IDs and prompts
            ids = [list(d.keys())[0] for d in dataset_loader.prompts]
            prompts = [list(d.values())[0] for d in dataset_loader.prompts]

            # Format responses and extract code
            no_inst = model_loader.format_responses(prompts, responses)
            only_code = dataset_loader.format_code_responses(no_inst)

            test_info = dataset_loader.test_code(ids, only_code)

            # Get tokens generated
            tokens_generated = model_loader.get_tokens_generated(no_inst)
            
            # Format patches
            formatted_patches = dataset_loader.format_patches(
                ids, prompts, only_code, tot_time, tokens_generated, test_info
            )

            # Append results for the current dataset
            model_result.append({dataset_loader.name: formatted_patches})

        # Append results for the current model
        evaluation_result.append({model_loader.name: model_result})

    # Return the final evaluation results
    return evaluation_result

if __name__ == "__main__":
    # Initialize configurator and get loaders
    conf = Configurator()
    model_loader = ModelLoader(conf)
    dataset_loaders = conf.get_dataset_loader()

    # Generate answers and evaluate
    json_data = evaluate([model_loader], dataset_loaders)

    # Write JSON to a file
    write_dict_to_json(json_data, conf.json_path)
