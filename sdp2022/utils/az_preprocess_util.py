import re

# import spacy
#
# nlp_lg = spacy.load('en_core_web_lg')

import nltk

ABBREVIATION_MAP = {
    "e.g.": "eg",
    "i.e.": "ie",
    "et al.": "et al",
    "etc.": "etc",
    "fig.": "figure",
    "cf.": "cf",
    "E.g.": "Eg",
    "Etc.": "Etc",
    "Fig.": "Figure",
    "Cf.": "Cf",
}

REJECTED_REGEX = {
    # 1- found in arXiv
    "arXiv:\d+\.\w+",
    "\[.+\.[A-Z]+\] \d\d* [A-Z][a-z][a-z] \d\d\d\d",
}


def remove_non_ascii(raw_text):
    chars = []
    for c in raw_text:
        if 127 >= ord(c) >= 1:
            chars.append(c)
    return "".join(chars)


def normalize_abbreviations(raw_text):
    for abbr in ABBREVIATION_MAP:
        raw_text = raw_text.replace(abbr, ABBREVIATION_MAP[abbr])
    return raw_text


def remove_non_necessary_hyphen(raw_text):
    chars = []
    text_len = len(raw_text)
    for i in range(text_len):
        if raw_text[i] == "-" and (
            i >= text_len - 1 or ("A" <= raw_text[i + 1] <= "Z")
        ):
            continue
        chars.append(raw_text[i])
    return "".join(chars)


def calculate_numeric_percent(text):
    return float(len("".join(re.findall("\d", text)))) / float(len(text))


def split_paragraph_sentences(par):
    return nltk.sent_tokenize(par)
    # doc = nlp_lg(par)
    # return [sent.text for sent in doc.sents]


def remove_rejected_regex(text):
    for regex in REJECTED_REGEX:
        text = re.sub(regex, "", text).strip()
    return text


def remove_non_informative_sentences(sentences):
    cleaned_sentences = []
    for sent in sentences:
        if calculate_numeric_percent(sent) > 0.25:
            continue
        if len(sent.split()) <= 5:
            continue
        cleaned_sentences.append(sent)
    return cleaned_sentences


def merge_not_completed_sentences(sentences):
    merged_sentences = []
    cur_merged_index = -1
    for i in range(len(sentences)):
        sent = sentences[i]
        if "a" <= sent[0] <= "z" and cur_merged_index > -1:
            merged_sentences[cur_merged_index] = (
                merged_sentences[cur_merged_index] + " " + sent
            )
        elif (
            cur_merged_index >= 0
            and "0" <= merged_sentences[cur_merged_index][-2] <= "9"
        ) and ("0" <= sent[0] <= "9" or len(sent.split()) < 4):
            merged_sentences[cur_merged_index] = (
                merged_sentences[cur_merged_index] + sent
            )
        else:
            cur_merged_index += 1
            merged_sentences.append(sent)
    return merged_sentences
