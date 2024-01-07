from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.generation import GenerationConfig
from typing import List, Tuple
import torch
import logging
import time
from utils import print_progress_bar
from dataset_loader.dataset_loader import DatasetLoader


class ModelLoader:
    def __init__(self, conf, model_id: str, template_name: str) -> None:
        self.model_id = model_id
        self.template_name = template_name

        self.cache_dir = conf.model_dir
        self.temperature = conf.temperature
        self.max_length = conf.max_length
        self.top_p = conf.top_p
        self.patch_size = conf.patches_per_bug
        self.batch_size = 1  # Initialize with a conservative batch size
        self.chat_template = ""

        self.device = "cuda"
        self.name = model_id.split("/")[1]

    def load_model_tokenizer(self):
        print("Loading of " + self.name + " model starting...")
        # Load model and tokenizer on GPU
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map=self.device,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            use_fast=True,
            device_map=self.device,
            cache_dir=self.cache_dir,
            chat_template=self.set_chat_template(self.template_name),
        )

        print("Loading of " + self.name + " model complete.\n")
        self.model, self.tokenizer = model, tokenizer
        
    
    def unload_model_tokenizer(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()

    def set_chat_template(self, template_name: str) -> str:
        content = ""
        with open("./prompt_templates/" + template_name, 'r') as file:
            for line in file:
                content += line.rstrip().lstrip()
        return content

    def generate_answers(
        self,
        dataset_loader: DatasetLoader,
    ) -> Tuple[List[List[str]], List[float]]:
        ret_responses = []
        ret_tot_time = []
        dataset_prompts = dataset_loader.prompts
        print("Generating answers for dataset: " + dataset_loader.name)

        for bug_nr, dataset_prompt in enumerate(dataset_prompts):
            prompt = list(dataset_prompt.values())[0]
            patches, tot_time = self.batch_completion(
                prompt, bug_nr, len(dataset_prompts)
            )
            # id = list(dataset_prompt.keys())[0]
            ret_responses.append(patches)
            ret_tot_time.append(tot_time)

        print("\n")
        return ret_responses, ret_tot_time

    def format_tabs(self, responses: List[List[str]]) -> List[List[str]]:
        formatted_responses = []

        for patches in responses:
            formatted_patches = []
            for patch in patches:
                formatted_response = patch.replace("\t", "    ")
                formatted_patches.append(formatted_response)
            formatted_responses.append(formatted_patches)

        return formatted_responses

    def remove_special_tokens(self, responses: List[List[str]]) -> List[List[str]]:
        modified_responses = []

        for response in responses:
            patches = [patch for patch in response]

            for token in self.tokenizer.all_special_tokens:
                patches = [patch.replace(token, "") for patch in patches]

            modified_responses.append(patches)

        return modified_responses

    def remove_inst(
        self, prompts: List[List[dict]], responses: List[List[str]]
    ) -> List[List[str]]:
        result = []

        for prompt, response_list in zip(prompts, responses):
            inst_enc = self.tokenizer.apply_chat_template(
                prompt, return_tensors="pt"
            ).to("cuda")
            inst = self.tokenizer.decode(inst_enc[0])

            updated_responses = [
                response.replace(inst, "") for response in response_list
            ]
            result.append(updated_responses)

        return result

    def format_responses(
        self, prompts: List[List[dict]], responses: List[List[str]]
    ) -> List[List[str]]:
        no_inst = self.remove_inst(prompts, responses)
        no_inst = self.remove_special_tokens(no_inst)
        no_inst = self.format_tabs(no_inst)
        return no_inst

    def get_tokens_generated(self, responses: List[List[str]]) -> List[float]:
        total_tokens = []
        for patches in responses:
            tokens_generated = 0
            for patch in patches:
                tokens_generated += len(
                    self.tokenizer.encode(patch, add_special_tokens=False)
                )
            total_tokens.append(tokens_generated)

        return total_tokens

    @torch.inference_mode()
    def batch_completion(
        self,
        prompt: List[dict],
        bug_nr: int,
        total_bugs: int,
    ) -> Tuple[List[str], float]:
        tot_time = 0
        input = None
        batch_completions = []
        gen_cfg = GenerationConfig.from_model_config(self.model.config)

        input = self.tokenizer.apply_chat_template(prompt, return_tensors="pt").to(
            self.device
        )

        generated_count = 0

        # Print progress
        print_progress_bar(
            (bug_nr * self.patch_size) + generated_count, total_bugs * self.patch_size
        )

        while generated_count < self.patch_size:
            try:
                with torch.no_grad():
                    batch_size = self.batch_size
                    if generated_count + self.batch_size > self.patch_size:
                        batch_size = self.patch_size - generated_count

                    start = time.time()
                    generated_ids = self.model.generate(
                        input,
                        use_cache=True,
                        generation_config=gen_cfg,
                        max_length=self.max_length,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        do_sample=True,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=batch_size,
                    )
                    tot_time += time.time() - start
                    batch_completions.extend(generated_ids)
                    generated_count += batch_size
                    # Check GPU memory usage
                    memory_allocated = torch.cuda.memory_allocated(self.device)

                    # Assume a buffer for safety
                    buffer_size = 2e9  # 2 GB
                    if memory_allocated < (
                        torch.cuda.get_device_properties(self.device).total_memory
                        - buffer_size
                    ):
                        # Double the batch size for the next iteration
                        self.batch_size *= 2

                    # Print progress
                    print_progress_bar(
                        (bug_nr * self.patch_size) + generated_count,
                        total_bugs * self.patch_size,
                    )

            except RuntimeError:
                self.batch_size = int(self.batch_size / 2)
                logging.info(
                    "GPU out of memory: Tried to run to may batches at once, halfing batch size and trying again."
                )

        responses = [self.tokenizer.decode(ids) for ids in batch_completions]

        return responses, tot_time
