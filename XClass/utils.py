import numpy as np
import os
import re

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

def tensor_to_numpy(tensor):
    return tensor.clone().detach().cpu().numpy()

def cosine_sim_embeddings(emb_a, emb_b):
    return np.dot(emb_a, np.transpose(emb_b)) / np.outer(np.linalg.norm(emb_a, axis=1), np.linalg.norm(emb_b, axis=1))

def cosine_sim_embedding(emb_a, emb_b):
    return np.dot(emb_a, emb_b) / np.linalg.norm(emb_a) / np.linalg.norm(emb_b)

def load(dataset_name):
    with open(os.path.join(dataset_name, 'dataset.txt'), mode='r', encoding='utf-8') as text_file:
        text = list(map(lambda x: x.strip(), text_file.readlines()))
    with open(os.path.join(dataset_name, 'classes.txt'), mode='r', encoding='utf-8') as classnames_file:
        class_names = "".join(classnames_file.readlines()).strip().split("\n")
    text = [s.strip() for s in text]

    cleaned_text = [clean_str(doc) for doc in text]

    result = {
        "class_names": class_names,
        "raw_text": text,
        "cleaned_text": cleaned_text,
    }
    return result