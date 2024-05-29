import pandas as pd
import subprocess
from tqdm import tqdm
import os

dataset_df = pd.read_csv("./dataset-info.csv")
print(dataset_df.head())

model = "MediaTek-Research/Breeze-7B-32k-Instruct-v1_0"


for _, dataset_row in dataset_df.iterrows():
    tqdm.write(
        f'Running dataset:{dataset_row["dataset_name"]} on model:{model}')

    subprocess.run(['python', 'translate.py', '-i', f'{os.path.join(os.path.dirname(__file__) ,dataset_row["path"] ) }',
                    '-d', f'../dataset/translate/{dataset_row["dataset_name"]}-en.csv', '-m', model])
