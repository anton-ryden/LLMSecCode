from typing import List, Dict

from configurator import Configurator
from dataset_loader.dataset_loader import DatasetLoader
from model_loader.model_loader import ModelLoader
from utils import save_json, print_progress_bar
from patch_tracker.dataset_store import DatasetStore
from patch_tracker.bug import Bug
from patch_tracker.patch import Patch


def evaluate_models_on_datasets(
    model_loaders: List[ModelLoader],
    dataset_loaders: List[DatasetLoader],
    configurator: Configurator,
):
    """
    Evaluate a list of model loaders on a list of dataset loaders.

    :param model_loaders: List of ModelLoader instances.
    :param dataset_loaders: List of DatasetLoader instances.
    :param configurator: Configurator instance.
    :return: List of dictionaries containing evaluation results for each model on each dataset.
    """
    evaluation_results = []

    # Iterate over each model loader
    for model_loader in model_loaders:
        model_loader.load_model_tokenizer()

        model_result = evaluate_single_model_on_datasets(
            model_loader, dataset_loaders, configurator
        )
        # Append results for the current model
        evaluation_results.append({model_loader.name: model_result})

        # Unload model and tokenizer from memory
        model_loader.unload_model_tokenizer()


def evaluate_single_model_on_datasets(
    model_loader: ModelLoader,
    dataset_loaders: List[DatasetLoader],
    configurator: Configurator,
) -> List[Dict]:
    """
    Evaluate a model loader on multiple dataset loaders.

    :param model_loader: Instance of ModelLoader.
    :param dataset_loaders: List of DatasetLoader instances.
    :param configurator: Configurator instance.
    :return: List of dictionaries containing evaluation results for the model on each dataset.
    """
    model_evaluation_results = []
    # Iterate over each dataset loader
    for dataset_loader in dataset_loaders:
        # Load prompts
        dataset_loader.load_prompts(
            configurator.max_chain_depth, configurator.patches_per_bug
        )

        # Create the objects to store the info
        dataset_store = DatasetStore(
            dataset_loader.name, configurator.max_chain_depth, dataset_loader.bugs
        )

        evaluate_single_model_on_dataset(
            model_loader, dataset_loader, dataset_store, configurator.max_chain_depth
        )

        dataset_store.update_stats()

        # Save results
        save_json(
            dataset_store.to_brief_summary_json(configurator),
            f"./results/{configurator.results_dir}/{model_loader.name}/{dataset_store.name}/brief_summary.json",
        )
        save_json(
            dataset_store.to_summary_json(configurator),
            f"./results/{configurator.results_dir}/{model_loader.name}/{dataset_store.name}/summary.json",
        )
        save_json(
            dataset_store.to_detailed_json(),
            f"./results/{configurator.results_dir}/{model_loader.name}/{dataset_store.name}/detailed.json",
        )

    return model_evaluation_results


def evaluate_single_model_on_dataset(
    model_loader: ModelLoader,
    dataset_loader: DatasetLoader,
    dataset_store: DatasetStore,
    max_chain_depth: int,
) -> None:
    """
    Evaluate a model loader on a single dataset loader.

    :param model_loader: Instance of ModelLoader.
    :param dataset_loader: Instance of DatasetLoader.
    :param dataset_store: Instance of DatasetStore.
    :param max_chain_depth: Maximum chain depth.
    """
    for depth in range(max_chain_depth):
        patches = []
        if depth == 0:
            for bug in dataset_store.bugs:
                patches.extend(bug.patches[0])
        else:
            patches = get_failed_patches(dataset_store.bugs, depth)
            model_loader.max_length = (depth+1) * model_loader.max_length

        generate_answers(patches, depth, dataset_loader, model_loader)

        test_answers(patches, depth, dataset_loader)

    print()


def generate_answers(
    patches: List[Patch],
    depth: int,
    dataset_loader: DatasetLoader,
    model_loader: ModelLoader,
):
    """
    Generate answers for a given set of patches.

    :param patches: List of patches.
    :param depth: Depth of the chain..
    :param dataset_loader: Instance of DatasetLoader.
    :param model_loader: Instance of ModelLoader.
    """
    if len(patches) == 0:
        return

    if depth == 0:
        print(f"Generating answers for dataset: {dataset_loader.name}")
    else:
        print(f"Generating answers Chain-Of-Thought nr: {depth}")
    print_progress_bar(0, len(patches))

    for i, patch in enumerate(patches, start=1):
        # Generate model responses and timing
        patch.llm_resp, patch.time_to_gen = model_loader.prompt_llm(patch.prompt)

        # Format responses and extract code
        patch.llm_resp_clean = model_loader.clean_response(patch.prompt, patch.llm_resp)
        patch.code = dataset_loader.extract_code(patch.llm_resp_clean)

        # Update tokens generated
        patch.tokens_generated = model_loader.get_tokens_generated(patch.llm_resp_clean)

        print_progress_bar(i, len(patches))
    print()


def test_answers(patches: List[Patch], depth: int, dataset_loader: DatasetLoader):
    """
    Test answers for a given set of patches.

    :param patches: List of patches.
    :param depth: Depth of the chain.
    :param dataset_loader: Instance of DatasetLoader.
    """
    if len(patches) == 0:
        return

    if depth == 0:
        print(f"Testing answers for dataset: {dataset_loader.name}")
    else:
        print(f"Testing answers Chain-Of-Thougth nr: {depth}")

    print_progress_bar(0, len(patches))
    for i, patch in enumerate(patches, start=1):
        dataset_loader.test_code(patch)

        print_progress_bar(i, len(patches))
    print()


def get_failed_patches(bugs: list[Bug], depth: int) -> List[Patch]:
    """
    Get failed patches for a given depth.

    :param bugs: List of bugs.
    :param depth: Depth of the chain.
    :return: List of failed patches.
    """
    patches = []
    for bug in bugs:
        for patch in bug.patches[depth - 1]:
            if patch.failed > 0 or patch.syntax_error or patch.other_error:
                next_patch = patch.get_next_chain()
                bug.add_patch(next_patch)
                patches.append(next_patch)

    return patches


if __name__ == "__main__":
    # Initialize configurator and get loaders
    configurator = Configurator()
    model_loaders = configurator.get_model_loaders()
    dataset_loaders = configurator.get_dataset_loaders()

    # Generate answers and evaluate
    evaluate_models_on_datasets(model_loaders, dataset_loaders, configurator)
