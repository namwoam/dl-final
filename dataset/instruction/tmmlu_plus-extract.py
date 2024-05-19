from datasets import load_dataset
import pandas as pd
import os
import csv

"""
task_list = [
    'engineering_math', 'dentistry', 'traditional_chinese_medicine_clinical_medicine', 'clinical_psychology', 'technical', 'culinary_skills', 'mechanical', 'logic_reasoning', 'real_estate',
    'general_principles_of_law', 'finance_banking', 'anti_money_laundering', 'ttqav2', 'marketing_management', 'business_management', 'organic_chemistry', 'advance_chemistry',
    'physics', 'secondary_physics', 'human_behavior', 'national_protection', 'jce_humanities', 'politic_science', 'agriculture', 'official_document_management',
    'financial_analysis', 'pharmacy', 'educational_psychology', 'statistics_and_machine_learning', 'management_accounting', 'introduction_to_law', 'computer_science', 'veterinary_pathology',
    'accounting', 'fire_science', 'optometry', 'insurance_studies', 'pharmacology', 'taxation', 'trust_practice', 'geography_of_taiwan', 'physical_education', 'auditing', 'administrative_law',
    'education_(profession_level)', 'economics', 'veterinary_pharmacology', 'nautical_science', 'occupational_therapy_for_psychological_disorders',
    'basic_medical_science', 'macroeconomics', 'trade', 'chinese_language_and_literature', 'tve_design', 'junior_science_exam', 'junior_math_exam', 'junior_chinese_exam',
    'junior_social_studies', 'tve_mathematics', 'tve_chinese_language', 'tve_natural_sciences', 'junior_chemistry', 'music', 'education', 'three_principles_of_people',
    'taiwanese_hokkien'
]
for task in task_list:
    val = load_dataset('ikala/tmmluplus', task)['validation']
    dev = load_dataset('ikala/tmmluplus', task)['train']
    test = load_dataset('ikala/tmmluplus', task)['test']
"""


high_school = ["junior_science_exam", "junior_chinese_exam",
               "junior_social_studies", "junior_chemistry", "jce_humanities"]
middle_school = ["tve_chinese_language", "tve_natural_sciences",
                 "traditional_chinese_medicine_clinical_medicine"]
other = ["geography_of_taiwan", "human_behavior", "politic_science"]

dataset = None

for tasks in [high_school, middle_school, other]:
    for task in tasks:
        train = load_dataset('ikala/tmmluplus', task)["test"].to_pandas()
        train["question"] = train["question"] + \
            "(A)"+train["A"]+"(B)"+train["B"]+"(C)"+train["C"]+"(D)"+train["D"]
        train["info"] = range(1, len(train) + 1)
        train["info"] = train["info"].apply(lambda x: f"TMMLU+-{task}-"+str(x))
        train["question"] = train["question"].apply(lambda x: x.replace("\n",""))
        train = train.drop(columns=["A", "B", "C", "D"])
        train["explanation"] = ""
        train = train[["info", "answer", "explanation", "question"]]
        if dataset is None:
            dataset = train
        else:
            dataset = pd.concat([dataset, train],
                                ignore_index=True, sort=False)

dataset.to_csv(os.path.join(os.path.dirname(__file__),
               "./tmmlu_plus.csv"), quotechar='"', quoting=csv.QUOTE_NONNUMERIC, index=False)
