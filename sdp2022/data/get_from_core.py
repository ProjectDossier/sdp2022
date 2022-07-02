import requests
import json

import pandas as pd
from time import sleep
from tqdm import tqdm


def pretty_json(json_object):
    print(json.dumps(json_object, indent=2))


test_df = pd.read_csv("../../data/raw/task1_test_no_label.csv")
train_df = pd.read_csv("../../data/raw/task1_train_dataset.csv")

with open("../../data/api_key.txt", "r") as apikey_file:
    api_key = apikey_file.readlines()[0].strip()
api_endpoint = "https://api.core.ac.uk/v3/"


def get_entity(url_fragment):
    headers={"Authorization":"Bearer "+api_key}
    response = requests.get(api_endpoint + url_fragment, headers=headers)
    if response.status_code == 200:
        return response.json(), response.elapsed.total_seconds()
    else:
        print(f"Error code {response.status_code}, {response.content}")
        return {}, None


outputs = []
for index_i, x in tqdm(enumerate(train_df['core_id'].tolist()), total=(len(train_df))):
    if (index_i+1) % 45 == 0:
        sleep(300)
    result = get_entity(f"outputs/{x}")[0]
    outputs.append(result)

df = pd.DataFrame.from_dict(outputs)

df.to_csv("out.csv")