import pandas as pd
import csv
files = ["112_chinese.csv", "111_chinese.csv",
         "110_chinese.csv", "109_chinese.csv",
         "112_social_studies.csv", "111_social_studies.csv",
         "110_social_studies.csv", "109_social_studies.csv"]

df = None
for file in files:
    local_df = pd.read_csv(file)
    local_df["question"] = local_df["question"].apply(lambda x:x.replace("\n","").replace("\r",""))
    local_df["explanation"] = local_df["explanation"].apply(lambda x:x.replace("\n","").replace("\r",""))
    if df is None:
        df = local_df
    else:
        df = pd.concat([df, local_df])
        

df.to_csv("./gsat-instruction.csv" , quotechar='"', quoting=csv.QUOTE_NONNUMERIC, index=False)