"""
python script that contain functions to collect the data set
"""

import argparse
import csv
from sdp2022.utils import io_util
import requests
import sys

csv.field_size_limit(sys.maxsize)


def download_articles_pdf(articles_info_file, output_directory):
    with open(articles_info_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            download_url = row['downloadUrl']
            paper_id = row['id']
            print("processing", paper_id)
            if not download_url:
                continue
            print("downloading", paper_id)
            download_pdf(download_url, io_util.join(output_directory, str(paper_id) + ".pdf"))


def download_pdf(download_url, download_path):
    if io_util.path_exits(download_path):
        return
    try:

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0',
        }
        r = requests.get(download_url, stream=True, allow_redirects=True, headers=headers)
        with open(download_path, 'wb') as f:
            for ch in r:
                f.write(ch)
    except:
        print("  Failed to open: " + download_url)


def main(args):
    info_file = args.info_file
    download_dir = args.download_dir

    download_articles_pdf(info_file, download_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Given CSV file of CORE data info, download pdfs of articles'
    )

    parser.add_argument('info_file', help='CSV file of the info about each paper and the download url')
    parser.add_argument('download_dir', help='The download dir of the pdfs')

    args = parser.parse_args()
    main(args)
