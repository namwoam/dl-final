
import transformers
import torch
from datasets import Dataset
import pandas as pd
import os
import re
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
        device_map="auto"
    )
    answers = []
    scores = []
    for question in tqdm(dataset,desc="Inference"):
        messages = [
            {"role": "system",
                "content": """你是一個用於解決臺灣高中生升學考試選擇題的 AI 助理，請依據邏輯推理及高中程度的知識選出正確的答案。
                           題目分為國文、英文、數學、自然、社會五個科目，題目可為單選題或多選題，若題目並未明文說明則皆視為單選題，只須在第一行輸出答案，不可輸出其他字元。
                           以下為輸出格式的範例：

                           水以固體、液體與氣體三相存在於地球系統中，相變時會伴隨著潛熱釋放或吸收。下列哪些現象會伴隨潛熱釋放？（應選三項）
                           (A)清晨時水氣凝結形成露珠時 (B)在高緯度地區冰直接變成水氣時
                           (C)地面積雪融化時 (D)水氣附著到凝結核上形成冰晶時
                           (E)夏季午後常見到的對流雲形成時
                           輸出：
                           ADE
                           """},
            {"role": "user", "content": question["question"]},
        ]

        tqdm.write("Question: "+question["info"])

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
            temperature=0.1,
            top_p=0.9,
        )
        tqdm.write("### LLM Response ###")
        tqdm.write(outputs[0]["generated_text"][len(prompt):])
        clean_answer = re.sub(r'[^A-Za-z0-9 ]+', '', outputs[0]["generated_text"][len(prompt):].split("\n")[0])
        answers.append(clean_answer)
        if len(question["answer"])==1:
            score = 1 if clean_answer == question["answer"] else 0
        else:
            correct_answers = set([ch for ch in question["answer"]])
            llm_answers = set([ch for ch in clean_answer])
            mismatch_count = len(correct_answers.union(llm_answers))-len(correct_answers.intersection(llm_answers))
            score = 1 - 0.4  * mismatch_count
            score = score if score > 0 else 0
        scores.append(score)
    df["score"]=scores
    df["generated_answer"]=answers
    df.to_csv("./baseline-result.csv")

if __name__ == "__main__":
    print(main())
