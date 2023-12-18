from typing import List, Dict
from configurator import Configurator
from dataset_loader.dataset_loader import DatasetLoader
from model_loader.model_loader import ModelLoader
from utils import write_dict_to_json


def evaluate_models_on_datasets(
    model_loaders: List[ModelLoader],
    dataset_loaders: List[DatasetLoader],
) -> List[dict]:
    """
    Evaluate a list of model loaders on a list of dataset loaders.

    :param model_loaders: List of ModelLoader instances.
    :param dataset_loaders: List of DatasetLoader instances.
    :return: List of dictionaries containing evaluation results for each model on each dataset.
    """
    # List to store the final evaluation results
    evaluation_results = []

    # Iterate over each model loader
    for model_loader in model_loaders:
        model_result = evaluate_single_model_on_datasets(model_loader, dataset_loaders)
        # Append results for the current model
        evaluation_results.append({model_loader.name: model_result})

    # Return the final evaluation results
    return evaluation_results


def evaluate_single_model_on_datasets(
    model_loader: ModelLoader,
    dataset_loaders: List[DatasetLoader],
) -> List[Dict]:
    """
    Evaluate a model loader on multiple dataset loaders.

    :param model_loader: Instance of ModelLoader.
    :param dataset_loaders: List of DatasetLoader instances.
    :return: List of dictionaries containing evaluation results for the model on each dataset.
    """
    # List to store results for each dataset
    model_evaluation_results = []

    # Iterate over each dataset loader
    for dataset_loader in dataset_loaders:
        formatted_dataset_result = evaluate_single_model_on_dataset(
            model_loader, dataset_loader
        )
        # Append results for the current dataset
        model_evaluation_results.append({dataset_loader.name: formatted_dataset_result})

    return model_evaluation_results


def evaluate_single_model_on_dataset(
    model_loader: ModelLoader,
    dataset_loader: DatasetLoader,
) -> List[Dict]:
    """
    Evaluate a model loader on a single dataset loader.

    :param model_loader: Instance of ModelLoader.
    :param dataset_loader: Instance of DatasetLoader.
    :return: List of dictionaries containing evaluation results for the model on the dataset.
    """
    # Generate model responses and timing
    responses, tot_time = model_loader.generate_answers(dataset_loader)

    # Get ids and prompt in lists
    ids, prompts = [], []
    for data_dict in dataset_loader.prompts:
        for key, value in data_dict.items():
            ids.append(key)
            prompts.append(value)

    # Format responses and extract code
    no_instruction = model_loader.format_responses(prompts, responses)
    only_code = dataset_loader.format_code_responses(no_instruction)

    test_info = dataset_loader.test_code(ids, only_code)

    # Get tokens generated
    tokens_generated = model_loader.get_tokens_generated(no_instruction)

    # Format patches
    formatted_patches = dataset_loader.format_patches(
        ids, prompts, only_code, tot_time, tokens_generated, test_info
    )

    return formatted_patches


if __name__ == "__main__":
    # Initialize configurator and get loaders
    conf = Configurator()
    model_loader = ModelLoader(conf)
    dataset_loaders = conf.get_dataset_loader()

    # Generate answers and evaluate
    json_data = evaluate_models_on_datasets([model_loader], dataset_loaders)

    # Write JSON to a file
    write_dict_to_json(json_data, conf.json_path)
