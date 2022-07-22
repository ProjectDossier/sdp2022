import argparse
import requests
from sdp2022.utils import io_util
from multiprocessing import Queue, Process


def worker(proc_num, input_files_queue, input_dir, output_dir, az_identification_endpoint):
    while True:
        if input_files_queue.empty():
            break
        pdf_file = input_files_queue.get()
        print(proc_num, "Input file.. ", pdf_file)
        paper_id = pdf_file[:-4]
        if io_util.path_exits(io_util.join(output_dir, paper_id + '.json')):
            continue
        az_identification_response = identify_az_for_article(input_dir, pdf_file, az_identification_endpoint)

        if az_identification_response.status_code == 200:
            io_util.write_json(io_util.join(output_dir, paper_id + '.json'), az_identification_response.json())
        else:
            print(proc_num, 'Failed to identify AZ for', pdf_file)


def identify_az_for_directory(input_path, output_path, az_identification_endpoint):
    pdf_files = [file for file in io_util.list_files_in_dir(input_path) if file.endswith('.pdf')]

    for pdf_file in pdf_files:
        print('processing', pdf_file)
        if io_util.path_exits(io_util.join(output_path, pdf_file[:-4] + '.json')):
            continue
        az_identification_response = identify_az_for_article(input_path, pdf_file, az_identification_endpoint)

        if az_identification_response.status_code == 200:
            io_util.write_json(io_util.join(output_path, pdf_file[:-4] + '.json'), az_identification_response.json())
        else:
            print('Failed to identify AZ for', pdf_file)


def identify_az_for_article(input_path, pdf_file, az_identification_endpoint):
    multipart_form_data = {
        'pdf_article': (pdf_file, open(io_util.join(input_path, pdf_file), "rb")),
    }
    az_identification_response = requests.post(az_identification_endpoint, verify=False,
                                               files=multipart_form_data, data={"paper_id": pdf_file[:-4]})
    return az_identification_response


def get_input_pdf_papers_queue(input_dir):
    pdf_files = [file for file in io_util.list_files_in_dir(input_dir) if file.endswith('.pdf')]
    input_pdf_papers_queue = Queue()
    for input_file in pdf_files:
        if input_file.startswith('.'):
            continue
        input_pdf_papers_queue.put(input_file)
    return input_pdf_papers_queue


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    az_identification_endpoint = args.az_identification_endpoint

    input_pdf_papers_queue = get_input_pdf_papers_queue(input_dir)

    procs = [Process(target=worker, args=[i, input_pdf_papers_queue, input_dir, output_dir, az_identification_endpoint])
             for i in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Given a directory of pdf articles, parse and identify argumentative zones of the articles '
    )

    parser.add_argument('input_dir', help='Input directory of pdfs articles ')
    parser.add_argument('output_dir', help='The output directory of the identified AZ')
    parser.add_argument('az_identification_endpoint', help='The end point of the AZ identification service.')

    args = parser.parse_args()
    main(args)
