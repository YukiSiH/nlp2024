import os
import random
import torch
import numpy as np
from torch.utils.data import TensorDataset
from transformers import InputExample, glue_convert_examples_to_features as convert_examples_to_features

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 获取类别标签
def get_labels(data_dir):
    with open(os.path.join(data_dir, 'classes.txt'), mode='r', encoding='utf-8') as classnames_file:
        class_names = "".join(classnames_file.readlines()).strip().split("\n")
    return list(range(len(class_names)))

# 加载数据样本
def load_examples(data_dir, dataset_name, suffix):
    dir_path = f"{dataset_name}_{suffix}" if suffix else dataset_name
    
    with open(os.path.join(data_dir, dir_path, 'dataset.txt'), mode='r', encoding='utf-8') as text_file:
        texts = [line.strip() for line in text_file.readlines()]
    
    with open(os.path.join(data_dir, dir_path, 'labels.txt'), mode='r', encoding='utf-8') as label_file:
        labels = [int(line.strip()) for line in label_file.readlines()]

    examples = [
        InputExample(guid=f"{suffix}-{i}", text_a=text, label=label)
        for i, (text, label) in enumerate(zip(texts, labels))
    ]
    return examples

# 加载和缓存数据集
def load_and_cache_examples(args, dataset_name, tokenizer, evaluate=False):
    suffix = "eval" if evaluate else "train"
    examples = load_examples(args.data_dir, dataset_name, suffix)
    
    label_list = get_labels(args.data_dir)
    output_mode = "classification"
    
    features = convert_examples_to_features(
        examples, tokenizer, max_length=args.max_seq_length, label_list=label_list, output_mode=output_mode
    )
    
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset
