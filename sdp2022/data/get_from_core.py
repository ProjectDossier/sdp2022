import requests
import json

import pandas as pd
from time import sleep
from tqdm import tqdm


def pretty_json(json_object):
    print(json.dumps(json_object, indent=2))


def get_entity(url_fragment):
    headers={"Authorization":"Bearer "+api_key}
    response = requests.get(api_endpoint + url_fragment, headers=headers)
    if response.status_code == 200:
        return response.json(), response.elapsed.total_seconds()
    elif response.status_code == 410:
        return {}, None
    elif response.status_code == 429:
        sleep(300)
        response = requests.get(api_endpoint + url_fragment, headers=headers)
        if response.status_code == 200:
            return response.json(), response.elapsed.total_seconds()
        else:
            return {}, None
    else:
        print(f"Error code {response.status_code}, {response.content}")
        return {}, None


with open("../../data/api_key.txt", "r") as apikey_file:
    api_key = apikey_file.readlines()[0].strip()
api_endpoint = "https://api.core.ac.uk/v3/"


if __name__ == '__main__':
    test_df = pd.read_csv("../../data/raw/task1_test_no_label.csv")
    train_df = pd.read_csv("../../data/raw/task1_train_dataset.csv")
    out_file = "../../data/processed/train_core.csv"

    outputs = []
    for index_i, x in tqdm(enumerate(train_df['core_id'].tolist()), total=(len(train_df))):
        result = get_entity(f"outputs/{x}")[0]
        result['original_index'] = index_i
        outputs.append(result)

        if (index_i + 1) % 600 == 0:
            sleep(400)

        if (index_i + 1) % 100 == 0:
            df = pd.DataFrame(outputs)
            df.to_csv(out_file)

    df = pd.DataFrame(outputs)
    df.to_csv(out_file)
