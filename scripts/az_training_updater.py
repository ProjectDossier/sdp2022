"""
python script to update the training files with the az information
"""

import argparse
import csv
from sdp2022.utils import io_util
import sys

csv.field_size_limit(sys.maxsize)


def update_with_az(input_file, output_file, az_info_directory):
    file = open(input_file)

    csvreader = csv.reader(file)
    header = next(csvreader)

    with open(output_file, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        header.extend(
            ['Claim_Abs', 'Method_Abs', 'Result_Abs', 'Conclusion_Abs', 'Claim', 'Method', 'Result', 'Conclusion'])
        writer.writerow(header)
        for row in csvreader:
            paper_id = str(row[11])
            # print(paper_id)
            az_fields = ['', '', '', '', '', '', '', '']
            if io_util.path_exits(io_util.join(az_info_directory, paper_id + '.json')):
                az_sent_map = get_az_sent_map(io_util.join(az_info_directory, paper_id + '.json'))
                az_fields = [az_sent_map['Claim_Abs'], az_sent_map['Method_Abs'], az_sent_map['Result_Abs'],
                             az_sent_map['Conclusion_Abs'], az_sent_map['Claim'], az_sent_map['Method'],
                             az_sent_map['Result'], az_sent_map['Conclusion']]
                print('found az data for', paper_id)
            elif io_util.path_exits(io_util.join(az_info_directory, paper_id + '.0.json')):
                az_sent_map = get_az_sent_map(io_util.join(az_info_directory, paper_id + '.0.json'))
                az_fields = [az_sent_map['Claim_Abs'], az_sent_map['Method_Abs'], az_sent_map['Result_Abs'],
                             az_sent_map['Conclusion_Abs'], az_sent_map['Claim'], az_sent_map['Method'],
                             az_sent_map['Result'], az_sent_map['Conclusion']]
                print('found az data for', paper_id)
            row.extend(az_fields)
            writer.writerow(row)


def get_az_sent_map(az_file_path):
    data = io_util.read_json(az_file_path)
    az_sent_map = {'Claim_Abs': '', 'Method_Abs': '', 'Result_Abs': '', 'Conclusion_Abs': '',
                   'Claim': '', 'Method': '', 'Result': '', 'Conclusion': ''}

    for section in data['sections']:
        for par in section['selected_sentences']:
            for sent in par['sentences']:
                if sent['tag'] not in az_sent_map:
                    continue
                key = sent['tag']
                if section['section_name'] == 'Abstract':
                    key = sent['tag'] + '_Abs'
                az_sent_map[key] = (az_sent_map[key] + ' ' + sent['sent']).strip()
    return az_sent_map


def main(args):
    input_file = args.input_file
    output_file = args.output_file
    az_dir = args.az_dir

    update_with_az(input_file, output_file, az_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Given CSV file of CORE data info and directory of json files with AZ information, update the csv file with az information'
    )

    parser.add_argument('input_file', help='CSV file of the info about each paper')
    parser.add_argument('output_file', help='CSV output path with updated AZ info')
    parser.add_argument('az_dir', help='The json files directory with AZ information.')

    args = parser.parse_args()
    main(args)
