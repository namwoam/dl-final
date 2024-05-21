from datasets import Dataset
import pandas as pd
import os
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import time
import csv
from sklearn.utils import shuffle

load_dotenv()

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "tmmlu_plus.csv"))
df = shuffle(df)
df = df.head(n=2000)
dataset = Dataset.from_pandas(df)

verbose = True

detailed_answers = []
for question in tqdm(dataset, desc="Generating Explanation"):
    time.sleep(2)  # prevent rate limit
    messages = [
        {"role": "system",
            "content": """你是一個用於解決臺灣高中生升學考試選擇題的 AI 助理，請依據邏輯推理及高中程度的知識選出正確的答案及回答正確且邏輯合理的解釋。                          """},
        {"role": "user", "content": f"""
                           以下為一題選擇題的題目及正確答案，請輸出該選擇該答案的解釋。
                           題目：{question["question"]}
                           答案：{question["answer"]}
                           
                           輸出格式：
                           正確的答案：{question["answer"]}
                           解釋：
                           [填入解釋]
                           """}
    ]

    outputs = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=512,
        temperature=0.0,
        messages=messages
    )
    if verbose:
        tqdm.write("Question: "+question["info"])
        tqdm.write("### Question ###")
        tqdm.write(question["question"])
        tqdm.write("### LLM Response ###")
        tqdm.write(outputs.choices[0].message.content)
        tqdm.write("### Answer ###")
        tqdm.write(question["answer"])
    generated_answer = outputs.choices[0].message.content
    generated_answer = "\n".join(list(filter(lambda x: len(
        x.replace(" ", "")) > 0, generated_answer.split("\n"))))
    detailed_answers.append(generated_answer.replace("\n", ""))

df["explanation"] = detailed_answers
df.to_csv(os.path.join(os.path.dirname(__file__), "tmmlu_plus-explain_1k.csv"), index=False, quotechar='"',
          quoting=csv.QUOTE_NONNUMERIC)
