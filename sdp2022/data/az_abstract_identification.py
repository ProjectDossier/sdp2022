import argparse
import csv
import requests

from sdp2022.utils import preprocess_util

categories_to_fields_map = {'claim': 'Claim_Abs', 'method': 'Method_Abs',
                            'result': 'Result_Abs', 'conclusion': 'Conclusion_Abs'}


def identify_az_for_abstracts(input_file, output_file, az_abstract_identification_endpoint):
    with open(input_file) as in_file:
        reader = csv.DictReader(in_file)
        headers = reader.fieldnames
        with open(output_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()

            for row in reader:
                print('processing', row['id'])
                if row['abstract']:
                    abstract_sentences = split_paragraph_sentences(row['abstract'])
                    print(abstract_sentences)
                    az_category_sent_map = classify_sentences_to_az(abstract_sentences, row['id'],
                                                                    az_abstract_identification_endpoint)
                    print(az_category_sent_map)
                    for category in az_category_sent_map:
                        row[category] = az_category_sent_map[category]
                    # writer.writerow(row)
                    # exit(0)
                writer.writerow(row)


def split_paragraph_sentences(paragraph):
    # 1. preprocess
    # a. clean non asci
    cleaned_paragraph = preprocess_util.remove_non_ascii(paragraph)

    cleaned_paragraph = preprocess_util.normalize_abbreviations(cleaned_paragraph)
    # c. remove hyphen
    cleaned_paragraph = preprocess_util.remove_non_necessary_hyphen(cleaned_paragraph)
    # d. clean rejected regex
    cleaned_paragraph = preprocess_util.remove_rejected_regex(cleaned_paragraph)

    # 2. split sentences
    sentences = preprocess_util.split_paragraph_sentences(cleaned_paragraph)

    # 3. post process sentences
    # a. remove non informative  sentences
    cleaned_sentences = preprocess_util.remove_non_informative_sentences(sentences)
    # b. merge uncomplete sentences
    cleaned_sentences = preprocess_util.merge_not_completed_sentences(cleaned_sentences)

    return cleaned_sentences


def classify_sentences_to_az(sentences, paper_id, az_abstract_identification_endpoint):
    response = requests.post(az_abstract_identification_endpoint,
                             verify=False,
                             json={'doc_id': str(paper_id), 'abstract_sentences': sentences})

    print(response.content)

    categories_sent_map = {'Claim_Abs': '', 'Method_Abs': '', 'Result_Abs': '', 'Conclusion_Abs': ''}

    if response.status_code != 200:
        print("failed to classify", paper_id)
        return categories_sent_map
    for sent in response.json()['abstract_sentences']:
        if sent['az_annotation'] in categories_to_fields_map:
            categories_sent_map[categories_to_fields_map[sent['az_annotation']]] \
                = (categories_sent_map[categories_to_fields_map[sent['az_annotation']]] + ' ' + sent['text']).strip()

    return categories_sent_map


def main(args):
    input_file = args.input_file
    output_file = args.output_file
    az_abstract_identification_endpoint = args.az_abstract_identification_endpoint

    identify_az_for_abstracts(input_file, output_file, az_abstract_identification_endpoint)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Given the csv file of training data information, identify AZ for abstract if found '
    )

    parser.add_argument('input_file', help='Input csv file of training info')
    parser.add_argument('output_file', help='The output csv file with abstract AZ information.')
    parser.add_argument('az_abstract_identification_endpoint',
                        help='The end point of the AZ abstract identification service.')

    args = parser.parse_args()
    main(args)
