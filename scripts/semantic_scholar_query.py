import argparse
import requests
import json
from tqdm import tqdm
import time
from os.path import join as join_path


API_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/"


def query_api(paper_id, field):
    response = requests.get(f"{API_ENDPOINT}{paper_id}/{field}")
    if response.status_code == 200:
        return response.json(), response.elapsed.seconds
    else:
        return {'data': []}, []


def semantic_scholar_mapping(out_path: str):
    with open(join_path(out_path, 'CORE_s_scholar_mapping_train.jsonl'), "r") as f_in:
        with open(join_path(out_path, 'references_citations_train.jsonl'), "w") as f_out:
            idx = 1
            for line in tqdm(f_in):
                idx += 2
                paper = json.loads(line)
                paper['references'] = []
                paper['citations'] = []
                references, _ = query_api(paper['paperId'], "references")
                citations, _ = query_api(paper['paperId'], "citations")
                if len(references['data']) > 0:
                    [paper['references'].append(i['citedPaper']) for i in references['data']]
                    if len(citations['data']) > 0:
                        [paper['citations'].append(i['citingPaper']) for i in citations['data']]
                    f_out.write(json.dumps(paper))
                    f_out.write('\n')
                if idx % 90 == 0:
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
