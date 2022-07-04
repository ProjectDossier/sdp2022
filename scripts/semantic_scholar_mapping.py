import argparse
import requests
import json
from sdp2022.data.batch_processing import BatchProcessing
from tqdm import tqdm
import time
from os.path import join as join_path


API_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/"


def query_api(query, limit=1):
    query = {"q": '+'.join(query.split()), "limit": limit}
    response = requests.get(f"{API_ENDPOINT}search?query={query}")
    if response.status_code == 200:
        return response.json(), response.elapsed.seconds
    else:
        return {'data': []}, []


def semantic_scholar_mapping(out_path: str):
    data = BatchProcessing().data
    with open(join_path(out_path, 's_scholar_ids.jsonl'), "w") as f:
        for idx, row in tqdm(data.iterrows()):
            query = row.title
            result, _ = query_api(query)
            if len(result['data']) > 0:
                result = result['data'][0]
                result['core_id'] = row.core_id
                result['core_title'] = row.title
                f.write(json.dumps(result))
                f.write('\n')
            if (idx + 1) % 90 == 0:
                time.sleep(300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='semantic scholar id mapping')
    parser.add_argument('-p',
                        dest='path',
                        type=str,
                        help='path to the folder that will contain the output jsonl',
                        default="../data/external/")
    path = parser.parse_args().path
    semantic_scholar_mapping(path)
