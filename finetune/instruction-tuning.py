import argparse
import os
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# check: https://huggingface.co/docs/trl/en/sft_trainer

def main(model_id: str, dataset_path: str, load_lora_path: str, store_lora_path: str, verbose=False):
    dataset = Dataset.from_csv(dataset_path)
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
            output_texts.append(text)
        return output_texts

    response_template = " ### Answer:"
    collator = DataCollatorForCompletionOnlyLM(
        response_template, tokenizer=tokenizer)


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
