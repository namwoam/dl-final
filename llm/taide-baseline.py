import transformers
import torch

model_id = "taide/Llama3-TAIDE-LX-8B-Chat-Alpha1"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# From 113年學測 社會科第59題
messages = [
    {"role": "system", "content": "你是一個來自台灣的AI助理，你的名字是 TAIDE，樂於以台灣人的立場幫助使用者，會用繁體中文回答問題。"},
    {"role": "user", "content": 
"""
日本治臺後，將女子教育納入學制系統，但受到傳統「女子無才便是德」的觀念限制，相較於臺灣男學生與日籍女學生，臺灣女童受教育的仍屬少數。初等
女子教育規劃的課程內容，以日語、裁縫科及家政科等課程為主。女子中等學校雖早在1897年某都市就成立「國語學校第一附屬學校女子分校」，但直到1920年
代後各地才陸續設立。女子就學率提升之後，開始有女醫師、看護婦、助產士、女教員等專門或半專門職業人士。不過，有學者指出，當時女性仍高度集中於
家政和照顧類有關的職業，而非平均分布於各職業類別。請問： 
根據題文，日治時期初等女子教育的課程設計理念應為何？ 
(A)為求男女平權而提高女童就學率 
(B)訓練醫師、教員等專門職業人才 
(C)提供殖民統治所需要之基層人員 
(D)培養具日本女性婦德的賢妻良母      
"""},
]

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
print(outputs[0]["generated_text"][len(prompt):])
