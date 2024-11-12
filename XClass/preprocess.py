import os
import re
from collections import Counter
import numpy as np

def clean_html(string):
    pattern = re.compile(r'&lt;.*?&gt;')
    while True:
        match = pattern.search(string)
        if not match:
            break
        string = string[:match.start()] + " " + string[match.end():]
    return string


def clean_str(string):
    string = clean_html(string)
    string = re.sub(r"[^A-Za-z0-9(),.!?\"\']", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def load_labels(data_dir):
    with open(os.path.join(data_dir, 'labels.txt'), mode='r', encoding='utf-8') as label_file:
        labels = list(map(lambda x: int(x.strip()), label_file.readlines()))
    return labels


def compute_statistics(text, name):
    num_documents = len(text)
    word_counts_per_doc = [len(doc.split(" ")) for doc in text]

    max_len = max(word_counts_per_doc)
    avg_len = np.average(word_counts_per_doc)
    std_len = np.std(word_counts_per_doc)

    print(f"\n--- Statistics for {name}: ---")
    print(f"Total documents: {num_documents}")
    print(f"Max document length: {max_len} words")
    print(f"Avg document length: {avg_len:.2f} words")
    print(f"Document length std. dev: {std_len:.2f} words")
    print("-------------------------------")



def load(dataset_name):
    with open(os.path.join(dataset_name, 'dataset.txt'), mode='r', encoding='utf-8') as text_file:
        text = list(map(lambda x: x.strip(), text_file.readlines()))
    with open(os.path.join(dataset_name, 'classes.txt'), mode='r', encoding='utf-8') as classnames_file:
        class_names = "".join(classnames_file.readlines()).strip().split("\n")
    text = [s.strip() for s in text]
    compute_statistics(text, "raw_txt")

    cleaned_text = [clean_str(doc) for doc in text]
    compute_statistics(cleaned_text, "cleaned_txt")

    result = {
        "class_names": class_names,
        "raw_text": text,
        "cleaned_text": cleaned_text,
    }
    return result


if __name__ == '__main__':
    data = load('agnews')
