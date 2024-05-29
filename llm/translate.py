
import transformers
import torch
from datasets import Dataset
import pandas as pd
import os
import re
from tqdm import tqdm
import argparse


def main(model_id: str, dataset_paths: list[str], destination_path: str, verbose: bool = False):
    df = None
    for dataset_path in dataset_paths:
        if df is None:
            df = pd.read_csv(dataset_path)
        else:
            df = pd.concat([df, pd.read_csv(dataset_path)], axis=1)
    dataset = Dataset.from_pandas(df)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        torch_dtype="auto",
        device_map="auto",
        model_kwargs={"load_in_8bit": True}
    )
    translated_questions = []
    for question in tqdm(dataset, desc="Inference"):
        messages = [
            {"role": "system",
                "content": """你是一個專精於中英翻譯的語言模型"""},
            {"role":"user", "content":f"""
                           將以下考試題目中的所有內容由中文翻譯成英文，不要寫出答案或解釋，並以純文字格式回答。\n
                           [小宜在書上讀到一段資料：「十七世紀以後，咖啡風行歐洲，帶動咖啡消費風氣， 使之成為大眾化飲料，歐洲各國其後也開始在殖民地大量種植咖啡。」依據此段 歷史敘述，小宜推斷當時的咖啡市場價格應該會下跌。依市場分析，其推論之依 據應為下列何者？(A)咖啡已經成為大眾化飲品，消費市場便容易發生短缺, (B)咖啡消費需求增加，但殖民地大量種植供給增加更大, (C)歐洲各國在殖民地種植咖啡後，同步刺激了消費成長, (D)政府預期咖啡價格將有大波動，採取了價格管制政策]\n
                           輸出格式：
                           Translated Question: [question in english]
                           """},
            {"role":"assistant","content":"Translated Question: Xiaoyi reads a passage in a book: 'After the 17th century, coffee became popular in Europe, driving a coffee consumption trend, making it a mass-market beverage. European countries then began to plant coffee extensively in their colonies.' Based on this historical account, Xiaoyi infers that the coffee market price at that time should have fallen. According to market analysis, her inference is most likely based on which of the following? (A) Since coffee has become a mass-market drink, the consumption market is prone to shortages. (B) Coffee consumption demand is increasing, but the increase in supply from large-scale planting in colonies is even greater. (C) After European countries planted coffee in their colonies, it also stimulated the growth of consumption. (D) The government anticipated significant fluctuations in coffee prices and implemented price control policies."},
            {"role": "user", "content": f"""
                           將以下考試題目中的所有內容由中文翻譯成英文，不要寫出答案或解釋，並以純文字格式回答。\n
                           ["""+question["question"]+"""]\n
                           輸出格式：
                           Translated Question: [question in english]
                           """},
        ]
        prompt = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        terminators = [
            pipeline.tokenizer.eos_token_id
        ]

        outputs = pipeline(
            prompt,
            max_new_tokens=512,
            eos_token_id=terminators,
            pad_token_id=pipeline.tokenizer.eos_token_id,
            do_sample=False
        )
        if verbose:
            tqdm.write("Question: "+question["info"])
            tqdm.write("### Question ###")
            tqdm.write(question["question"])
            tqdm.write("### LLM Response ###")
            tqdm.write(outputs[0]["generated_text"][len(prompt):])
        generated_question = outputs[0]["generated_text"][len(prompt):]
        generated_question = "\n".join(list(filter(lambda x: len(
            x.replace(" ", "")) > 0, generated_question.split("\n"))))
        translated_questions.append(generated_question.replace("\n", " "))
    result_df = df[["info", "answer"]]
    result_df.loc[:, "translated_questions"] = translated_questions
    result_df.to_csv(destination_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", default=False,
                        action="store_true")
    parser.add_argument('-m', '--model_id',
                        nargs='?', default="taide/Llama3-TAIDE-LX-8B-Chat-Alpha1", help="model_id from huggingface")
    parser.add_argument('-d', '--destination_path',
                        nargs='?', default=os.path.join(os.path.dirname(__file__),
                                                        "baseline-result.csv"), help="path to store the .csv file result")
    parser.add_argument('-i', '--dataset_paths', nargs='+', default=[os.path.join(os.path.dirname(__file__),
                                                                                  "baseline-en-question.csv")], help="path for the testing dataset(s)")
    args = parser.parse_args()
    main(model_id=args.model_id, dataset_paths=args.dataset_paths,
         destination_path=args.destination_path, verbose=args.verbose)
