
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
    for question in tqdm(dataset, desc="Inference"):
        messages = [
            {"role": "system",
                "content": """你是一個用於解決臺灣高中生升學考試選擇題的 AI 助理，請依據邏輯推理及高中程度的知識選出正確的答案。
                           題目分為國文、英文、數學、自然、社會五個科目，題目可為單選題或多選題，若題目並未明文說明則皆視為單選題。
                           以下為輸出格式的範例：

                           全球主要有三大地震帶，臺灣位於其中的「環太平洋地震帶」上。下列有關此地震帶的敘述何者正確？
                           (A)此地震帶的形成主要與張裂性板塊邊界有關
                           (B)地震主要發生在地殼中，所以此地震帶特徵多淺源地震
                           (C)此地震帶與環太平洋火山帶（火環）位置幾乎一致，有許多活火山
                           (D)地震與斷層活動息息相關，此地震帶的地震多半是由平移斷層活動造成
                           (E)臺灣位在此地震帶上，表示臺灣島與太平洋板塊相接
                           輸出：
                           answer:{ADE}
                           解釋：
                           (A)環太平洋地震帶主要為聚合型板塊邊界，例如臺灣、馬里亞納海溝、日本。
                           (B)由於是聚合型板塊邊界，板塊有隱沒作用，地震震源應該由淺到深都有。
                           (D)承(B)，聚合擠壓作用為主的地區，其應力會以壓力為主，斷層多為逆斷層。
                           (E)位於此地震帶不等於與太平洋板塊相接，臺灣位於歐亞板塊及菲律賓海板塊的交界。
                           """},
            {"role": "user", "content": "題目為："+question["question"]},
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
            do_sample=False
        )
        tqdm.write("### LLM Response ###")
        tqdm.write(outputs[0]["generated_text"][len(prompt):])
        clean_answer = re.sub(
            r'[^A-Z]+', '', outputs[0]["generated_text"][len(prompt):].split("\n")[0])
        answers.append(clean_answer)
        if len(question["answer"]) == 1:
            score = 1 if clean_answer == question["answer"] else 0
        else:
            correct_answers = set([ch for ch in question["answer"]])
            llm_answers = set([ch for ch in clean_answer])
            mismatch_count = len(correct_answers.union(
                llm_answers))-len(correct_answers.intersection(llm_answers))
            score = 1 - 0.4 * mismatch_count
            score = score if score > 0 else 0
        scores.append(score)
    df["score"] = scores
    df["generated_answer"] = answers
    df.to_csv("./baseline-result.csv")


if __name__ == "__main__":
    print(main())
