from typing import List, Dict
from configurator import Configurator
from dataset_loader.dataset_loader import DatasetLoader
from model_loader.model_loader import ModelLoader
from utils import write_dict_to_json


def evaluate_models_on_datasets(
    model_loaders: List[ModelLoader],
    dataset_loaders: List[DatasetLoader],
    results_dir: str,
):
    """
    Evaluate a list of model loaders on a list of dataset loaders.

    :param model_loaders: List of ModelLoader instances.
    :param dataset_loaders: List of DatasetLoader instances.
    :param results_dir: Str name of folder to save results to.
    :return: List of dictionaries containing evaluation results for each model on each dataset.
    """
    evaluation_results = []

    # Iterate over each model loader
    for model_loader in model_loaders:
        model_loader.load_model_tokenizer()
        model_result = evaluate_single_model_on_datasets(model_loader, dataset_loaders)
        # Append results for the current model
        evaluation_results.append({model_loader.name: model_result})

        # Write model results to json file
        write_dict_to_json(
            {model_loader.name: model_result},
            "./results/" + results_dir + "/" + model_loader.name + ".json",
        )

        # Unload model and tokenizer from memory
        model_loader.unload_model_tokenizer()


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

    test_result = dataset_loader.test_code(ids, only_code)

    # Get tokens generated
    tokens_generated = model_loader.get_tokens_generated(no_instruction)

    # Format patches
    formatted_patches = dataset_loader.format_patches(
        no_instruction, ids, prompts, only_code, tot_time, tokens_generated, test_result
    )

    return formatted_patches


if __name__ == "__main__":
    # Initialize configurator and get loaders
    configurator = Configurator()
    model_loaders = configurator.get_model_loaders()
    dataset_loaders = configurator.get_dataset_loaders()

    # Generate answers and evaluate
    evaluate_models_on_datasets(model_loaders, dataset_loaders, configurator.results_dir)
