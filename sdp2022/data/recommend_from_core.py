import requests

import pandas as pd
from time import sleep
from tqdm import tqdm


with open("../../data/api_key.txt", "r") as apikey_file:
    api_key = apikey_file.readlines()[0].strip()
api_endpoint = "https://api.core.ac.uk/v3/"


def find_recommended(data):
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + api_key}
    response = requests.post(
        url=f"https://api.core.ac.uk/v3/recommend", headers=headers, json=data
    )

    if response.status_code == 200:
        return response.json(), response.elapsed.total_seconds()
    elif response.status_code == 410:
        return {}, None
    elif response.status_code == 429:
        sleep(300)
        response = requests.post(
            url=f"https://api.core.ac.uk/v3/recommend", headers=headers, json=data
        )
        if response.status_code == 200:
            return response.json(), response.elapsed.total_seconds()
        else:
            return {}, None
    else:
        print(f"Error code {response.status_code}, {response.content}")
        return {}, None


if __name__ == "__main__":
    input_file = "../../data/raw/task1_train_dataset.csv"
    out_file = "../../data/processed/train_recommended_core.csv"

    in_df = pd.read_csv(input_file)

    outputs = []
    for index_i, row in tqdm(in_df.iterrows(), total=len(in_df)):

        data = {
            "text": "string",
            "limit": "5",
            "abstract": row["title"],
        }
        result = find_recommended(data=data)[0]
        for paper in result:
            paper["original_index"] = index_i
            paper["original_title"] = row["title"]
            paper["original_id"] = row["core_id"]
            paper["theme"] = row["theme"]
            outputs.append(paper)

        if (index_i + 1) % 1000 == 0:
            sleep(300)

        if (index_i + 1) % 100 == 0:
            df = pd.DataFrame(outputs)
            df.to_csv(out_file)

    df = pd.DataFrame(outputs)
    df.to_csv(out_file)
