
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
    answers = []
    scores = []
    detailed_answers = []
    for question in tqdm(dataset, desc="Inference"):
        messages = [
            {"role": "system",
                "content": """You are an AI assistant designed to solve multiple-choice questions for Taiwanese university entrance exams. Please use logical reasoning and knowledge at the high school level to select the correct answer.。                           """},
            {"role": "user", "content": """
                           Could you help me answer the university entrance examination questions? The questions are divided into five subjects: Chinese, English, Mathematics, Science, and Social Studies. The questions can be single-choice or multiple-choice.



Example one:



There are three major earthquake zones globally, and Taiwan is located on the "Pacific Ring of Fire." Which of the following statements about this earthquake zone are correct? This question is a multiple-choice question.

(A) The formation of this earthquake zone is mainly related to extensional plate boundaries.

(B) Earthquakes mainly occur in the Earth's crust, so this earthquake zone features many shallow-focus earthquakes.

(C) This earthquake zone is almost consistent with the Pacific Ring of Fire (Ring of Fire) and has many active volcanoes.

(D) Earthquakes are closely related to fault movements; most earthquakes in this zone are caused by transform faults.

(E) Taiwan is located on this earthquake zone, indicating that the island of Taiwan is connected to the Pacific Plate.



Output format:

Correct answer: [fill in the correct option(s)]

Explanation:

[fill in explanation]
                           """},
            {"role": "assistant", "content": """
                           correct answers: (A), (D), (E)

                           Explanation:

                           (A) The Ring of Fire is mainly composed of convergent plate boundaries, such as Taiwan, Mariana Trench, and Japan.

                           (B) Since it is a convergent plate boundary, subduction occurs, and earthquake foci should range from shallow to deep.

                           (D) Referencing (B), regions dominated by convergent compressive forces will mainly experience stress in the form of pressure, and faults will mostly be reverse faults.

                           (E) Located in this seismic belt does not mean it is connected to the Pacific Plate; Taiwan is situated at the boundary between the Eurasian Plate and the Philippine Sea Plate.

                           Therefore, choose (A), (D), (E)
                           
             """},
            {"role": "user", "content": f"Next, please select the correct option(s) for the following question. This question is a {'single-choice question' if len(question['answer'])==1 else 'multiple-choice questio'}. Here is the question: "+question["translated_questions"]+"?"+"""

Output format:

correct answer(s): [Enter the correct option(s)]

Explanation:

[Enter the explanation]
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
            tqdm.write(question["translated_questions"])
            tqdm.write("### LLM Response ###")
            tqdm.write(outputs[0]["generated_text"][len(prompt):])
            tqdm.write("### Answer ###")
            tqdm.write(question["answer"])
        generated_answer = outputs[0]["generated_text"][len(prompt):]
        generated_answer = "\n".join(list(filter(lambda x: len(
            x.replace(" ", "")) > 0, generated_answer.split("\n"))))
        detailed_answers.append(generated_answer.replace("\n", ""))
        clean_answer = re.findall(
            r"\([A-Z]\)", generated_answer.split("\n")[0])
        clean_answer = "".join(clean_answer)
        clean_answer = re.sub(
            r'[^A-Z]+', '', clean_answer)
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
    result_df = df[["info", "answer"]]
    result_df.loc[:, "score"] = scores
    result_df.loc[:, "generated_answer"] = answers
    result_df.loc[:, "detailed_generated_answer"] = detailed_answers
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
                                                                                  "baseline-question.csv")], help="path for the testing dataset(s)")
    args = parser.parse_args()
    main(model_id=args.model_id, dataset_paths=args.dataset_paths,
         destination_path=args.destination_path, verbose=args.verbose)
