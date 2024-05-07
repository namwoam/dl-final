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
    for question in tqdm(dataset,desc="Inference"):

        messages = [
            {"role": "system",
                "content": """你是一個用於解決臺灣高中生升學考試選擇題的 AI 助理，題目分為國文、英文、數學、自然、社會五個科目，題目可為單選題或多選題，若題目並未明文說明則皆視為單選題。請在第一行輸出英文字母的答案，並在下一行輸出題目的題解。
			   ### 題目 ###
                           新聞曾報導在地球南極大陸發現來自火星的隕石，科學家何以推測該隕石來自火星？
                           (A)已經有載人太空船登陸火星，並曾攜回火星岩石
                           (B)太陽系早期火星與地球曾發生碰撞，可判斷當時有大量火星岩石掉落到地球
                           (C)經過化學分析，隕石中的元素同位素比例符合火星的元素同位素比例
                           (D)該隕石含有大量氧化鐵，呈現暗紅色
                           (E)該隕石外觀焦黑，有火燒的痕跡
                           ### 答案 ###
                           C
                           ### 題解 ###
                           分析各選項：
                           (A)目前仍未有載人太空船登陸火星，僅月球有人類登陸過
                           (B)落到地球表面的岩石已經經過各種地質作用影響，無法代表它起源的原始性質
                           (C)天體的同位素比例會和它的形成過程及材料有關
                           (D)地球地表也有氧化鐵，非火星獨有特徵
                           (E)隕石焦黑會發生在很多來源的隕石上，無法視作火星獨有的特性。
                           因此選(C)
			   """},
            {"role": "user", "content": question["question"]},
        ]

        tqdm.write("### Question ###")
        tqdm.write(question["question"])

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
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        tqdm.write("### LLM response ###")
        tqdm.write(outputs[0]["generated_text"][len(prompt):])

        tqdm.write("### Ground Truth ###")
        tqdm.write(question["answer"])


if __name__ == "__main__":
    print(main())
