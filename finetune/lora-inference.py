
from transformers import AutoModelForCausalLM,BitsAndBytesConfig,AutoTokenizer
import torch
from datasets import Dataset
import pandas as pd
import os
import re
from tqdm import tqdm
import argparse
from peft import PeftModel
from peft import prepare_model_for_kbit_training


def main(model_id:str , dataset_paths:list[str] , destination_path:str , lora_path:str,verbose:bool=False ):
    df = None
    for dataset_path in dataset_paths:
        if df is None:
            df = pd.read_csv(dataset_path)
        else:
            df = pd.concat([df, pd.read_csv(dataset_path)], axis=1)
    dataset = Dataset.from_pandas(df)
    
    
    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
    device_map = {"": 0}
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    original_model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                      device_map=device_map,
                                                      quantization_config=bnb_config,
                                                      trust_remote_code=True,
                                                      use_auth_token=True)
    original_model = prepare_model_for_kbit_training(original_model)
    peft_model = PeftModel.from_pretrained(original_model, lora_path,torch_dtype=torch.float16,is_trainable=False)
    peft_model.eval()
    answers = []
    scores = []
    detailed_answers = []
    for question in tqdm(dataset, desc="Inference"):
        messages = [
            {"role": "system",
                "content": """你是一個用於解決臺灣高中生升學考試選擇題的 AI 助理，請依據邏輯推理及高中程度的知識選出正確的答案。                           """},
            {"role": "user", "content": """
                           請你幫我回答高中的學測題目，題目分為國文、英文、數學、自然、社會五個科目，題目可為單選題或多選題。
                           範例一：

                           全球主要有三大地震帶，臺灣位於其中的「環太平洋地震帶」上。下列有關此地震帶的敘述何者正確？此題為多選題，
                           (A)此地震帶的形成主要與張裂性板塊邊界有關
                           (B)地震主要發生在地殼中，所以此地震帶特徵多淺源地震
                           (C)此地震帶與環太平洋火山帶（火環）位置幾乎一致，有許多活火山
                           (D)地震與斷層活動息息相關，此地震帶的地震多半是由平移斷層活動造成
                           (E)臺灣位在此地震帶上，表示臺灣島與太平洋板塊相接
                           
                           輸出格式：
                           正確的答案：[填入正確的選項]
                           解釋：
                           [填入解釋]
                           """},
            {"role":"assistant","content":"""
                           正確的答案：(A)、(D)、(E)
                           解釋：
                           (A)環太平洋地震帶主要為聚合型板塊邊界，例如臺灣、馬里亞納海溝、日本。
                           (B)由於是聚合型板塊邊界，板塊有隱沒作用，地震震源應該由淺到深都有。
                           (D)承(B)，聚合擠壓作用為主的地區，其應力會以壓力為主，斷層多為逆斷層。
                           (E)位於此地震帶不等於與太平洋板塊相接，臺灣位於歐亞板塊及菲律賓海板塊的交界。
                           因此選(A)(D)(E)
                           
             """},
            {"role": "user", "content": f"""
                           接下來請針對以下問題，選出正確的選項，此題為{"單選題" if len(question["answer"])==1 else "多選題"}。請問"""+question["question"]+"？"+"""
                           輸出格式：
                           正確的答案：[填入正確的選項]
                           解釋：
                           [填入解釋]
                           """},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt , return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = peft_model.generate(input_ids=inputs["input_ids"], max_new_tokens=2048)
            peft_model_output = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
        

        
        if verbose:
            tqdm.write("Question: "+question["info"])
            tqdm.write("### Question ###")
            tqdm.write(question["question"])
            tqdm.write("### LLM Response ###")
            tqdm.write(peft_model_output[len(prompt):])
            tqdm.write("### Answer ###")
            tqdm.write(question["answer"])
        generated_answer = peft_model_output[len(prompt):]
        generated_answer = "\n".join(list(filter(lambda x: len(x.replace(" ",""))>0  , generated_answer.split("\n"))))
        detailed_answers.append(generated_answer.replace("\n",""))
        clean_answer = re.sub(
            r'[^A-Z]+', '', generated_answer.split("\n")[0])
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
    result_df = df[["info","answer"]]
    result_df.loc[:,"score"] = scores
    result_df.loc[:,"generated_answer"] = answers
    result_df.loc[:,"detailed_generated_answer"] = detailed_answers
    result_df.to_csv(destination_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v","--verbose", help="increase output verbosity",default=False,
                    action="store_true")
    parser.add_argument('-m', '--model_id', 
                    nargs='?', default="MediaTek-Research/Breeze-7B-32k-Instruct-v1_0",help="model_id from huggingface")
    parser.add_argument('-d', '--destination_path', 
                    nargs='?', default=os.path.join(os.path.dirname(__file__),
                                  "./lora-result.csv"), help="path to store the .csv file result")
    parser.add_argument('-i', '--dataset_paths', nargs='+', default=[os.path.join(os.path.dirname(__file__),
                                  "../llm/baseline-question.csv")] , help="path for the testing dataset(s)")
    parser.add_argument('-l', '--lora_path', nargs='?', default=os.path.join(os.path.dirname(__file__),
                                  "cp-all_textbook/checkpoint-1700") , help="path for the lora file")
    args = parser.parse_args()
    main(model_id=args.model_id ,dataset_paths=args.dataset_paths,destination_path=args.destination_path,verbose=args.verbose , lora_path=args.lora_path)
    
