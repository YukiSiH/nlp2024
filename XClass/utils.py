import numpy as np
import os
import re
from sklearn.metrics import confusion_matrix, f1_score

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


def evaluate_predictions(true_class, predicted_class, output_to_console=True, return_tuple=False):
    confusion = confusion_matrix(true_class, predicted_class)
    if output_to_console:
        print("-" * 80 + "Evaluating" + "-" * 80)
        print(confusion)
    f1_macro = f1_score(true_class, predicted_class, average='macro')
    f1_micro = f1_score(true_class, predicted_class, average='micro')
    if output_to_console:
        print("F1 macro: " + str(f1_macro))
        print("F1 micro: " + str(f1_micro))
    if return_tuple:
        return confusion, f1_macro, f1_micro
    else:
        return {
            "confusion": confusion.tolist(),
            "f1_macro": f1_macro,
            "f1_micro": f1_micro
        }
