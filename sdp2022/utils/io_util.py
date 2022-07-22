"""
Contains some utils functions for io operations.
"""

import json
import os
import pickle


def path_exits(path):
    return os.path.exists(path)


def mkdir(path):
    if not path_exits(path):
        os.mkdir(path)


def list_files_in_dir(dir):
    return [file for file in os.listdir(dir) if is_file(join(dir, file))]


def list_directories(dir):
    return [subdir for subdir in os.listdir(dir) if os.path.isdir(join(dir, subdir))]


def is_file(path):
    return os.path.isfile(path)


def join(path1, path2):
    return os.path.join(path1, path2)


def write_json(path, dict):
    with open(path, 'w') as outfile:
        json.dump(dict, outfile, indent=2)
    outfile.close()


def read_json(path):
    with open(path, "r") as infile:
        data = json.load(infile)
    infile.close()
    return data


def read_file_into_list(input_file):
    lines = []
    with open(input_file, "r") as infile_fp:
        for line in infile_fp.readlines():
            lines.append(line.strip())
    infile_fp.close()
    return lines


def write_list_to_file(output_file, list):
    with open(output_file, "w") as outfile_fp:
        for line in list:
            outfile_fp.write(line + "\r\n")
    outfile_fp.close()


def write_text_to_file(output_file, text):
    with open(output_file, "w") as output_fp:
        output_fp.write(text)
    output_fp.close()


def write_pickle(data, file_path):
    pickle.dump(data, open(file_path, "wb"))


def read_pickle(file_path):
    return pickle.load(open(file_path, 'rb'))
