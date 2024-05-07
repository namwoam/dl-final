import transformers
import torch
from datasets import Dataset
import pandas as pd
import os
from tqdm import tqdm

model_id = "taide/Llama3-TAIDE-LX-8B-Chat-Alpha1"


def main():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                  "sample_questions.csv"), index_col=0)
    dataset = Dataset.from_pandas(df)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    for question in tqdm(dataset):

        messages = [
            {"role": "system",
                "content": "你是一個來自台灣的AI助理，你的名字是 TAIDE，樂於以台灣人的立場幫助使用者，會用繁體中文回答問題。"},
            {"role": "user", "content": question["question"]},
        ]

        print("### Question ###")
        print(question["question"])

        prompt = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        print("### LLM response ###")
        print(outputs[0]["generated_text"][len(prompt):])

        print("### Ground Truth ###")
        print(question["answer"])


if __name__ == "__main__":
    print(main())
