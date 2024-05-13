import pandas as pd
import subprocess
from tqdm import tqdm
import os
model_df = pd.read_csv("./model-info.csv")
print(model_df.head())

dataset_df = pd.read_csv("./dataset-info.csv")
print(dataset_df.head())

for _ , model_row in model_df.iterrows():
    for _ , dataset_row in dataset_df.iterrows():
        tqdm.write(f'Running dataset:{dataset_row["dataset_name"]} on model:{model_row["short_name"]}')
        if model_row["open"] is False:
            subprocess.run(['python', 'inference-claude.py', '-i',f'{os.path.join(os.path.dirname(__file__) ,dataset_row["path"] ) }' ,'-d',f'./{dataset_row["dataset_name"]}-result-{model_row["short_name"]}.csv' ,'-m' ,f'{model_row["model"]}'])
        else:
            subprocess.run(['python', 'inference.py', '-i',f'{os.path.join(os.path.dirname(__file__) ,dataset_row["path"] ) }' ,'-d',f'./{dataset_row["dataset_name"]}-result-{model_row["short_name"]}.csv' ,'-m' ,f'{model_row["model"]}'])
        
        