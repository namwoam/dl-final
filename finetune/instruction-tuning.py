import argparse
import os
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

# check: https://huggingface.co/docs/trl/en/sft_trainer


def main(model_id: str, dataset_path: str, load_lora_path: str, store_lora_path: str, verbose=False):
    dataset = Dataset.from_csv(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Instruction dataset size:{len(dataset)}")

    def format_row_as_instruction_prompt(example):
        # Check if 'input' key exists and has content

        # Define the prompts based on the presence of input
        primer_prompt = ("Below is an instruction that describes a task, paired with an input "
                         "that provides further context. Write a response that appropriately completes the request.")
        input_template = f"""###題目: 請針對以下問題，選出正確的選項，此題為{"單選題" if len(example['answer'])==1 else "多選題"}。請問"""+example["question"]+"？\n\n"

        instruction_template = f"""### 指令: 你是一個用於解決臺灣高中生升學考試選擇題的 AI 助理，請依據邏輯推理及高中程度的知識選出正確的答案。\n
                        輸出格式：\n
                        ### 正確的答案：[填入正確的選項]\n
                        ### 解釋：[填入解釋]\n\n"""

    # Check if 'output' key exists

        response_template = f"""### 正確的答案：{example["answer"]}\n
                                ### 解釋：{example["explanation"]}\n\n
                        """

        return f"{primer_prompt}\n\n{instruction_template}{input_template}{response_template}"

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    device_map = {"": 0}
    original_model = AutoModelForCausalLM.from_pretrained(model_id,
                                                          device_map=device_map,
                                                          quantization_config=bnb_config,
                                                          trust_remote_code=True,
                                                          use_auth_token=True)

    original_model = prepare_model_for_kbit_training(original_model)

    config = LoraConfig(
        r=32,  # Rank
        lora_alpha=32,
        target_modules=[
            'q_proj',
            'k_proj',
            'v_proj',
            'up_proj',
            'down_proj',
            'o_proj',
            'dense'
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )

    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    original_model.gradient_checkpointing_enable()

    peft_model = get_peft_model(original_model, config)

    max_seq_len = 512
    
    train_args = TrainingArguments(
        output_dir=store_lora_path,
        num_train_epochs=20,
        # trying to max out resources on colab
        per_device_train_batch_size=1,
        gradient_accumulation_steps=10,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=100,
        save_strategy="steps",
        save_steps=100,
        learning_rate=3e-5,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        disable_tqdm=False
    )

    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=dataset,
        max_seq_length=max_seq_len,
        tokenizer=tokenizer,
        packing=True,
        formatting_func=format_row_as_instruction_prompt,
        args=train_args,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", default=False,
                        action="store_true")
    parser.add_argument('-m', '--model_id',
                        nargs='?', default="MediaTek-Research/Breeze-7B-32k-Instruct-v1_0", help="model_id from huggingface")
    parser.add_argument('-i', '--dataset_paths', nargs='?',
                        default="../dataset/instruction/112_chinese.csv", help="path for the instruction dataset")
    parser.add_argument('-ll', '--load_lora_path', nargs='?', default=os.path.join(os.path.dirname(__file__),
                                                                                   "cp-all_textbook/"), help="path for the previous lora file")
    parser.add_argument('-sl', '--store_lora_path', nargs='?', default=os.path.join(os.path.dirname(__file__),
                                                                                    "is-all_gsat/"), help="path for the saving lora file")
    args = parser.parse_args()
    main(model_id=args.model_id, dataset_path=args.dataset_paths, verbose=args.verbose,
         load_lora_path=args.load_lora_path, store_lora_path=args.store_lora_path)
