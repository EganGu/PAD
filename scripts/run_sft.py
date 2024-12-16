#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random
import sys
import sys
sys.path.append('/hpc2hdd/JH_DATA/share/zrao538/PrivateShareGroup/zrao538_NLPGroup/yggu/SimPO')

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed

from alignment import (
    DataArguments,
    SFTConfig,
    H4ArgumentParser,
    ModelArguments,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from alignment.data import maybe_insert_system_message, is_openai_format
from peft import PeftConfig, PeftModel
# from simpo_trainer import SimPOTrainer
# from simpo_config import SimPOConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from dataclasses import dataclass, field
from typing import Optional, Literal

logger = logging.getLogger(__name__)

MISTRAL_CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"


def process(row, tokenizer):
    responses_chat = []

    # prompt_chat = tokenizer.apply_chat_template(row['chosen'][:-1], tokenize=False, add_generation_prompt=True)
    # prompt = tokenizer.apply_chat_template(
    #     [{'user'}], tokenize=False, add_generation_prompt=True)
    # input = tokenizer.apply_chat_template(
    #     [{'user'}, {'assistant'}], tokenize=False)
    # label = input[len(prompt)] = -100
    # # assert response_chat[:len(prompt_chat)] == prompt_chat
    # # response_chat = response_chat[len(prompt_chat):]

    # row['prompt'] = prompt_chat
    # row['responses'] = responses_chat
    row['chat'] = tokenizer.apply_chat_template(row['chosen'], tokenize=False)
    return row

def formatting_prompts_func(row):
    return row['chat']

def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig))
    model_args, data_args, training_args = parser.parse()

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["messages", "chosen", "rejected", "prompt", "completion", "label", "responses", "scores", "scores_mcq", "scores_logp", "scores_avglogp"],
        local=data_args.local_dataset,
    )
    if training_args.debug:
        for key in raw_datasets:
            raw_datasets[key] = raw_datasets[key].select(range(100))
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    # column_names = list(raw_datasets["train"].features)

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)

    # if "mistral" in model_args.model_name_or_path.lower():
    #     change_template = "mistral"
    # else:
    #     change_template = None
    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        process,
        fn_kwargs={
            "tokenizer": tokenizer,
        },
        num_proc=data_args.preprocessing_num_workers,
        desc="Formatting comparisons with prompt template",
    )
    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    # for split in ["train", "test"]:
    #     raw_datasets[split] = raw_datasets[split].rename_columns(
    #         {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
    #     )

    # Log a few random samples from the training set:
    # for index in random.sample(range(len(raw_datasets["train"])), 3):
    #     logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
    #     # logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
    #     # logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")
    #     logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['responses']}")

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        attn_implementation=model_args.attn_implementation,
    )

    model = model_args.model_name_or_path
    # seems to require internet
    # if is_adapter_model(model, model_args.model_revision) is True:
    #     logger.info(f"Loading SFT adapter for {model_args.model_name_or_path=}")
    #     peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
    #     model_kwargs = dict(
    #         revision=model_args.base_model_revision,
    #         trust_remote_code=model_args.trust_remote_code,
    #         use_flash_attention_2=model_args.use_flash_attention_2,
    #         torch_dtype=torch_dtype,
    #         use_cache=False if training_args.gradient_checkpointing else True,
    #         device_map=get_kbit_device_map() if quantization_config is not None else None,
    #         quantization_config=quantization_config,
    #     )
    #     base_model = AutoModelForCausalLM.from_pretrained(
    #         peft_config.base_model_name_or_path,
    #         **model_kwargs,
    #     )
    #     model = PeftModel.from_pretrained(
    #         base_model,
    #         model_args.model_name_or_path,
    #         revision=model_args.model_revision,
    #     )
    #     model_kwargs = None

    training_args.model_init_kwargs = model_kwargs
    #########################
    # Instantiate SimPO trainer
    #########################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
        formatting_func=formatting_prompts_func,
        data_collator=DataCollatorForCompletionOnlyLM('<start_of_turn>model\n', tokenizer=tokenizer),
        dataset_kwargs={"add_special_tokens": False}
    )

    ###############
    # Training loop
    ###############
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete! ***")


if __name__ == "__main__":
    main()
