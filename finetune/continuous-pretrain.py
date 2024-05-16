import os
from os import listdir
from os.path import isfile, join

from transformers import AutoModelForCausalLM , AutoTokenizer , DataCollatorForLanguageModeling
import pandas as pd
from datasets import Dataset

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

import torch
from  datetime import datetime
import pandas as pd
from peft import prepare_model_for_kbit_training , LoraConfig, get_peft_model

import argparse

def main(model_id ,dataset_paths,destination_path,verbose=False):
    ## Prepare tokenized dataset ##
    all_files = []
    for data_dir in dataset_paths:
        data_dir = "../dataset/textbook/ch"
        files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
        files = [f for f in files if f.endswith(".txt")]
        all_files.extend(files)
    
    sentence_length = 30
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    raw_dataset = []
    for file in all_files:
        file_content = open(join(data_dir, file)  , "r").readlines()
        for line in file_content:
            for start in range(0 , len(line) , sentence_length):
                raw_dataset.append({"text":line[start: min(start+sentence_length , len(line))]})
    
    dataset = Dataset.from_list(raw_dataset)
    dataset = dataset.train_test_split(test_size=0.01)


    def preprocess_function(examples):
        return tokenizer([" ".join(x) for x in examples["text"]])

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=dataset["train"].column_names,
    )

    total_token = 0
    for example in tokenized_dataset["train"]:
        total_token += len(example["input_ids"])
    print("Training token count:",total_token)
    
    block_size = 1024

    def group_texts(examples):
    # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
        result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=4)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    ## Prepare Model ##
    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
    
    device_map = {"": 0}
    original_model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                      device_map=device_map,
                                                      quantization_config=bnb_config,
                                                      trust_remote_code=True,
                                                      use_auth_token=True)
    
    original_model = prepare_model_for_kbit_training(original_model)
    
    config = LoraConfig(
        r=32, #Rank
        lora_alpha=32,
        target_modules=[
            'q_proj',
            'k_proj',
            'v_proj',
            'dense'
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )

    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    original_model.gradient_checkpointing_enable()

    peft_model = get_peft_model(original_model, config)
    
    peft_training_args = TrainingArguments(
        output_dir = destination_path,
        warmup_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        optim="paged_adamw_8bit",
        logging_steps=25,
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=25,
        evaluation_strategy="steps",
        eval_steps=25,
        do_eval=True,
        gradient_checkpointing=True,
        report_to="none",
        overwrite_output_dir = 'True',
        group_by_length=True,
        num_train_epochs=1
    )

    peft_model.config.use_cache = False
    peft_trainer = Trainer(
        model=peft_model,
        train_dataset= lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
        args=peft_training_args,
        data_collator=data_collator,
    )
    
    peft_trainer.train()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v","--verbose", help="increase output verbosity",default=False,
                    action="store_true")
    parser.add_argument('-m', '--model_id', 
                    nargs='?', default="taide/Llama3-TAIDE-LX-8B-Chat-Alpha1",help="model_id from huggingface")
    parser.add_argument('-d', '--destination_path', 
                    nargs='?', default=os.path.join(os.path.dirname(__file__),
                                  "baseline-result.csv"), help="path to store the .csv file result")
    parser.add_argument('-i', '--dataset_paths', nargs='+', default=[os.path.join(os.path.dirname(__file__),
                                  "baseline-question.csv")] , help="path for the testing dataset(s)")
    args = parser.parse_args()
    main(model_id=args.model_id ,dataset_paths=args.dataset_paths,destination_path=args.destination_path,verbose=args.verbose)
    
