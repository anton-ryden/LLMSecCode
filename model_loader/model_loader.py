from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.generation import GenerationConfig
from typing import List, Tuple
import torch
import logging
import time


class ModelLoader:
    """
    Class for loading and managing language model instances.
    """

    def __init__(self, conf, model_id: str, template_name: str) -> None:
        """
        Initialize the ModelLoader.

        :param conf: Configuration object.
        :param model_id: Identifier for the model.
        :param template_name: Name of the template file.
        """
        self.model_id = model_id
        self.template_name = template_name

        self.cache_dir = conf.model_dir
        self.temperature = conf.temperature
        self.max_length = conf.max_length
        self.top_p = conf.top_p
        self.patch_size = conf.patches_per_bug
        self.batch_size = 1
        self.chat_template = ""

        self.device = "cuda"
        self.name = model_id.split("/")[1]

    def load_model_tokenizer(self):
        """
        Load the model and tokenizer.
        """
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
        """
        Unload the model and tokenizer.
        """
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()

    def set_chat_template(self, template_name: str) -> str:
        """
        Set the chat template.

        :param template_name: Name of the template file.
        :return: Content of the template file.
        """
        content = ""
        with open("./prompt_templates/" + template_name, "r") as file:
            for line in file:
                content += line.rstrip().lstrip()
        return content

    def remove_special_tokens(self, no_inst: str) -> str:
        """
        Remove special tokens from the provided string.

        :param no_inst: The string from which to remove special tokens.
        :return: The string without special tokens.
        """
        for token in self.tokenizer.all_special_tokens:
            no_inst.replace(token, "")
        return no_inst

    def remove_inst(
        self,
        prompt: List[dict],
        no_inst: str,
    ) -> str:
        """
        Remove instruction from the provided string.

        :param prompt: List of dictionaries representing the prompt.
        :param no_inst: The string from which to remove instruction.
        :return: The string without instruction.
        """
        inst_enc = self.tokenizer.apply_chat_template(prompt, return_tensors="pt").to(
            "cuda"
        )
        inst = self.tokenizer.decode(inst_enc[0])
        updated_responses = no_inst.replace(inst, "")
        return updated_responses

    def clean_response(self, prompt: List[dict], llm_resp: str) -> str:
        """
        Clean the model response.

        :param prompt: List of dictionaries representing the prompt.
        :param llm_resp: Model response.
        :return: Cleaned response.
        """
        no_inst = self.remove_inst(prompt, llm_resp)
        no_inst = self.remove_special_tokens(no_inst)
        no_inst = no_inst.replace("\t", "    ")
        no_inst = no_inst.replace("\\n", "\n")
        no_inst = no_inst.strip()
        return no_inst

    def get_tokens_generated(self, clean_resp: str) -> int:
        """
        Get the number of tokens generated.

        :param clean_resp: The cleaned model response.
        :return: Number of tokens generated.
        """
        return len(self.tokenizer.encode(clean_resp, add_special_tokens=False))

    @torch.inference_mode()
    def prompt_llm(
        self,
        prompt: List[dict],
    ) -> Tuple[str, float]:
        """
        Prompt the language model.

        :param prompt: List of dictionaries representing the prompt.
        :return: Tuple containing the model response and total time taken.
        """
        tot_time = 0
        batch_completions = []
        gen_cfg = GenerationConfig.from_model_config(self.model.config)

        input = self.tokenizer.apply_chat_template(prompt, return_tensors="pt").to(
            self.device
        )

        try:
            with torch.no_grad():
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
                )
                tot_time += time.time() - start
                batch_completions.extend(generated_ids)
                torch.cuda.empty_cache()

        except Exception as e:
            logging.info(e)

        response = self.tokenizer.decode(batch_completions[0])

        return response, tot_time
