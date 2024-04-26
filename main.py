from typing import List, Dict
import time
import json
import os

from configurator import Configurator
from dataset_loader.dataset_loader import DatasetLoader
from model_loader.model_loader import ModelLoader
from utils.framework_utils import save_json, print_progress_bar, ROOT_PATH
from data_structures.dataset_store import DatasetStore
from data_structures.task import Task
from data_structures.answer import Answer

import sys
sys.path.append(f'{ROOT_PATH}/datasets/suites')

from PurpleLlama.CybersecurityBenchmarks.benchmark.run import main as cyberseceval_run
from PurpleLlama.CybersecurityBenchmarks.benchmark.llm import create as cyberseceval_create_llm
from PurpleLlama.CybersecurityBenchmarks.benchmark.llm import ANY

def evaluate_models_on_datasets(
    model_loaders: List[ModelLoader],
    dataset_loaders: List[DatasetLoader],
    configurator: Configurator,
) -> List[Dict]:
    """
    Evaluate a list of model loaders on a list of dataset loaders.

    Args:
        model_loaders (List[ModelLoader]): List of ModelLoader instances.
        dataset_loaders (List[DatasetLoader]): List of DatasetLoader instances.
        configurator (Configurator): Configurator instance.

    Returns:
        List[Dict]: List of dictionaries containing evaluation results for each model on each dataset.
    """
    evaluation_results = []

    # Iterate over each model loader
    for model_loader in model_loaders:
        model_loader.load_model_tokenizer()

        if configurator.run_cyberseceval:
            run_cyberseceval(model_loader)

        if dataset_loaders:
            model_result = evaluate_single_model_on_datasets(
                model_loader, dataset_loaders, configurator
            )
        else:
            return
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

    Args:
        model_loader (ModelLoader): Instance of ModelLoader.
        dataset_loaders (List[DatasetLoader]): List of DatasetLoader instances.
        configurator (Configurator): Configurator instance.

    Returns:
        List[Dict]: List of dictionaries containing evaluation results for the model on each dataset.
    """
    model_evaluation_results = []
    # Iterate over each dataset loader
    for dataset_loader in dataset_loaders:
        start = time.time()
        # Load prompts
        dataset_loader.load_prompts()

        tasks = dataset_loader.prompts.get_tasks(
            model_loader.conversation_type,
            model_loader.template_name,
            configurator.max_chain_depth,
            configurator.answers_per_task,
        )

        if len(tasks) == 0:
            print(
                f"NOTE: Skipping dataset {dataset_loader.name} with conversation type {model_loader.conversation_type} since it is not supported."
            )
            continue

        # Create the objects to store the info
        dataset_store = DatasetStore(
            dataset_loader.name,
            configurator.max_chain_depth,
            tasks,
            dataset_loader.area,
        )

        evaluate_single_model_on_dataset(
            model_loader,
            dataset_loader,
            dataset_store,
            configurator.max_chain_depth,
        )

        dataset_store.update_stats()

        run_time = time.time() - start
        print(
            f"Time for generating and testing {dataset_loader.name}: {round(run_time, 1)}s\n"
        )

        # Save results
        save_json(
            dataset_store.to_brief_summary_json(
                configurator,
                model_loader.conversation_type,
                model_loader.name,
                run_time,
            ),
            f"./results/{configurator.results_dir}/{model_loader.name}/{dataset_store.name}/{model_loader.conversation_type}_brief_summary.json",
        )
        save_json(
            dataset_store.to_summary_json(
                configurator,
                model_loader.conversation_type,
                model_loader.name,
                run_time,
            ),
            f"./results/{configurator.results_dir}/{model_loader.name}/{dataset_store.name}/{model_loader.conversation_type}_summary.json",
        )
        save_json(
            dataset_store.to_detailed_json(),
            f"./results/{configurator.results_dir}/{model_loader.name}/{dataset_store.name}/{model_loader.conversation_type}_detailed.json",
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

    Args:
        model_loader (ModelLoader): Instance of ModelLoader.
        dataset_loader (DatasetLoader): Instance of DatasetLoader.
        dataset_store (DatasetStore): Instance of DatasetStore.
        max_chain_depth (int): Maximum chain depth.
    """
    for depth in range(max_chain_depth):
        answers = []
        if depth == 0:
            for task in dataset_store.tasks:
                answers.extend(task.answers[0])
        else:
            answers = get_incorrect_answers(dataset_store.tasks, depth)

        generate_answers(answers, depth, dataset_loader, model_loader)

        stat = test_answers(answers, depth, dataset_loader, model_loader)

        dataset_store.add_stat(depth, stat)


def generate_answers(
    answers: List[Answer],
    depth: int,
    dataset_loader: DatasetLoader,
    model_loader: ModelLoader,
):
    """
    Generate answers for a given set of answers.

    Args:
        answers (List[Answer]): List of answers.
        depth (int): Depth of the chain.
        dataset_loader (DatasetLoader): Instance of DatasetLoader.
        model_loader (ModelLoader): Instance of ModelLoader.
    """
    if len(answers) == 0:
        return

    if depth == 0:
        print(f"Generating answers for dataset: {dataset_loader.name}")
    else:
        print(f"Generating answers Chain-Of-Thought depth: {depth}")
    print_progress_bar(0, len(answers))

    for i, answer in enumerate(answers, start=1):
        # Generate model responses and timing
        answer.llm_resp, answer.time_to_gen, answer.memory = model_loader.prompt_llm(
            answer.prompt_instance.prompt
        )

        # Format responses and extract code
        answer.llm_resp_clean = model_loader.clean_response(
            answer.prompt_instance.prompt, answer.llm_resp
        )

        answer.extract_code(model_loader.template_name)

        # Update tokens generated
        answer.tokens_generated = model_loader.get_tokens_generated(
            answer.llm_resp_clean
        )

        print_progress_bar(i, len(answers))
    print()


def test_answers(
    answers: List[Answer], depth: int, dataset_loader: DatasetLoader, model: ModelLoader
):
    """
    Test answers for a given set of answers.

    Args:
        answers (List[Answer]): List of answers.
        depth (int): Depth of the chain.
        dataset_loader (DatasetLoader): Instance of DatasetLoader.
    """
    if len(answers) == 0:
        return

    if depth == 0:
        print(f"Testing answers for dataset: {dataset_loader.name}")
    else:
        print(f"Testing answers Chain-Of-Thougth depth: {depth}")

    return dataset_loader.test_code(answers, model)


def get_incorrect_answers(tasks: list[Task], depth: int) -> List[Answer]:
    """
    Get answers that failed at least one test or had some kind of error for a given depth.

    Args:
        tasks (List[Task]): List of tasks.
        depth (int): Depth of the chain.

    Returns:
        List[Answer]: List of failed answers.
    """
    answers = []
    for task in tasks:
        for answer in task.answers[depth - 1]:
            if answer.failed > 0 or answer.syntax_error or answer.other_error:
                next_answer = answer.get_next_chain()
                task.add_answer(next_answer)
                answers.append(next_answer)

    return answers

def run_cyberseceval(model_loader: ModelLoader):
    model_name = f"{model_loader.name}_{model_loader.conversation_type}"
    model = ANY(model_name, "123", model_loader)
    cyberseceval_config_file = os.path.join(ROOT_PATH, "config", "cyberseceval_config.json")

    with open(cyberseceval_config_file, "r") as f:
        cyberseceval_configs = json.load(f)

    os.environ['WEGGLI_PATH'] = cyberseceval_configs["paths"]["WEGGLI_PATH"]
    os.environ['PATH'] = os.environ.get('WEGGLI_PATH', '') + ':' + os.environ.get('PATH', '')
    cyberseceval_dataset_path = os.path.join(ROOT_PATH, 'datasets', 'suites', 'PurpleLlama', 'CybersecurityBenchmarks', 'datasets')
    results_path = os.path.join(ROOT_PATH, 'results', cyberseceval_configs["testing_configs"]["results_dir"], model_name)

    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    benchmarks = cyberseceval_configs["testing_configs"]["benchmarks"]
    llm_under_test = [model]
    judge_llm = None
    expansion_llm = None

    for benchmark in benchmarks:
        
        if benchmark == "mitre":
            judge_llm = cyberseceval_configs["benchmark_configs"][benchmark]["judge_llm"]
            expansion_llm = cyberseceval_configs["benchmark_configs"][benchmark]["expansion_llm"]
            judge_host, judge_name, _ = judge_llm.split("::")
            if judge_host == "ANY":
                judge_model = create_framework_model("judge", cyberseceval_configs)
                judge_llm = ANY(judge_name, "123", judge_model)
            else:
                judge_llm = cyberseceval_create_llm(judge_llm)
            expansion_host, expansion_name, _ = expansion_llm.split("::")
            if expansion_host == "ANY":
                expansion_model = create_framework_model("expansion", cyberseceval_configs)
                expansion_llm = ANY(expansion_name, "123", expansion_model)
            else:
                expansion_llm = cyberseceval_create_llm(expansion_llm)
            if cyberseceval_configs["benchmark_configs"]["mitre"]["with_augmentation"]:
                prompt_path = os.path.join(cyberseceval_dataset_path, f"{benchmark}/mitre_benchmark_100_per_category_with_augmentation.json")
            else:
                prompt_path = os.path.join(cyberseceval_dataset_path, f"{benchmark}/mitre_benchmark_100_per_category.json")
        else:
            prompt_path = os.path.join(cyberseceval_dataset_path, f"{benchmark}/{benchmark}.json")

        if benchmark == "prompt-injection":
            prompt_path = os.path.join(cyberseceval_dataset_path, f"prompt_injection/prompt_injection.json")
            judge_llm = cyberseceval_configs["benchmark_configs"][benchmark]["judge_llm"]
            judge_host, judge_name, _ = judge_llm.split("::")
            if judge_host == "ANY":
                judge_model = create_framework_model("judge", cyberseceval_configs)
                judge_llm = ANY(judge_name, "123", judge_model)
            else:
                judge_llm = cyberseceval_create_llm(judge_llm)

        if benchmark == "interpreter":
            judge_llm = cyberseceval_configs["benchmark_configs"][benchmark]["judge_llm"]
            judge_host, judge_name, _ = judge_llm.split("::")
            if judge_host == "ANY":
                judge_model = create_framework_model("judge", cyberseceval_configs)
                judge_llm = ANY(judge_name, "123", judge_model)
            else:
                judge_llm = cyberseceval_create_llm(judge_llm)

        if benchmark == "canary-exploit":
            expansion_llm = cyberseceval_configs["benchmark_configs"][benchmark]["judge_llm"]
            expansion_host, expansion_name, _ = expansion_llm.split("::")
            if expansion_host == "ANY":
                expansion_model = create_framework_model("expansion", cyberseceval_configs)
                expansion_llm = ANY(expansion_name, "123", expansion_model)
            else:
                expansion_llm = cyberseceval_create_llm(expansion_llm)
        
        response_path = os.path.join(results_path, f"{benchmark}_responses.json")
        judge_response_path = os.path.join(results_path, f"{benchmark}_judge_responses.json")
        stat_path = os.path.join(results_path, f"{benchmark}_stat.json")
    
        print(f"\nStarting {benchmark} benchmark with CyberSecEval2...\n")

        if judge_llm is not None:
            print(f"    Judge LLM: {judge_llm.model}\n")
        if expansion_llm is not None:
            print(f"    Expansion LLM: {expansion_llm.model}\n")

        cyberseceval_run(default_benchmark=benchmark, 
                        llms_under_test=llm_under_test,
                        default_prompt_path=prompt_path,
                        default_response_path=response_path,
                        default_stat_path=stat_path,
                        default_judge_response_path=judge_response_path,
                        judge_llm=judge_llm,
                        expansion_llm=expansion_llm)
        
        if judge_llm is not None:
            judge_model.unload_model_tokenizer()
        if expansion_llm is not None:
            expansion_model.unload_model_tokenizer()

class ConfigurationProxy:
    def __init__(self):
        self.__dict__ = {}

    def __getattr__(self, name):
        return self.__dict__.get(name)

    def __setattr__(self, name, value):
        self.__dict__[name] = value

def create_framework_model(type: str, config: dict) -> ModelLoader:
    configuration = ConfigurationProxy()
    for key, value in config["testing_configs"][f"{type}_llm_config"].items():
            setattr(configuration, key, value)
    model_id, template_name, conversation_type = configuration.model_config.split(":")
    new_model = ModelLoader(configuration, model_id, template_name, conversation_type)
    new_model.load_model_tokenizer()

    return new_model



if __name__ == "__main__":
    # Initialize configurator and get loaders
    configurator = Configurator()
    model_loaders = configurator.get_model_loaders()
    dataset_loaders = configurator.get_dataset_loaders()

    # Generate answers and evaluate
    evaluate_models_on_datasets(model_loaders, dataset_loaders, configurator)
