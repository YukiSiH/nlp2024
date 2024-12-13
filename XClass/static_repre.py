import os
import string
from tqdm import tqdm
import torch
import numpy as np
from collections import Counter
import pickle as pk
from datasets import load
from transformers import BertModel, BertTokenizer
from utils import tensor_to_numpy, load

vocab_min_occurrence = 5
layer = 12

def prepare_tokenized_chunks(tokenizer, text):
    max_model_tokens = 512 
    max_tokens = max_model_tokens - 2  # 包含 [CLS]和[SEP]
    sliding_window_size = max_tokens // 2  # 滑动窗口大小，通常为最大token数的一半

    if not hasattr(prepare_tokenized_chunks, "sos_token_id"):
        # 如果尚未设置特殊token的id，则初始化它们
        prepare_tokenized_chunks.sos_token_id, prepare_tokenized_chunks.eos_token_id = tokenizer.encode("", add_special_tokens=True)

    tokenized_words = tokenizer.basic_tokenizer.tokenize(text, never_split=tokenizer.all_special_tokens) # 特殊符号不拆分

    tokenid_pos = []  # 存储token id和它们在chunk中的位置
    chunk_list = []  # chunk列表
    current_chunk = []  # 子词分词后的所有id放入current_chunk中

    # 会把wordpiece分词后的结果按块存储，处理长文本
    for index, word in enumerate(tokenized_words + [None]):
        if word is not None:
            word_tokens = tokenizer.wordpiece_tokenizer.tokenize(word)  # wordpiece tokenizer 子词分词

        if word is None or len(current_chunk) + len(word_tokens) > max_tokens: # 一块存满了或者处理完所有的词
            # 将当前chunk加到chunks列表中
            chunk_list.append([prepare_tokenized_chunks.sos_token_id] + current_chunk + [prepare_tokenized_chunks.eos_token_id])
            # 应用滑动窗口，保留最后一部分token，保留一定的上下文信息
            current_chunk = current_chunk[-sliding_window_size:] if sliding_window_size > 0 else []

        # 记录每个token在chunk列表中的位置
        # (chunk_id, start_pos, end_pos)
        if word is not None:
            tokenid_pos.append(
                (len(chunk_list), len(current_chunk), len(current_chunk) + len(word_tokens))
            )
            # 子词分词后的所有id放入current_chunk中
            current_chunk.extend(tokenizer.convert_tokens_to_ids(word_tokens))

    return tokenized_words, tokenid_pos, chunk_list


# 结合了块内的上下文。
def process_embeddings(model, layer, tokenized_text, tokenid_pos, chunk_list):
    layer_embeddings = []   # 存储每个分块的层嵌入
    for tokenids_chunk in chunk_list:
        input_ids = torch.tensor([tokenids_chunk], device=model.device)
        with torch.no_grad():
            hidden_states = model(input_ids)
        all_layer_outputs = hidden_states[2]  # 获取模型的所有层输出

        # 提取指定层的嵌入，移除开头和结尾的特殊标记 [CLS] 和 [SEP]
        layer_embedding = tensor_to_numpy(all_layer_outputs[layer].squeeze(0))[1: -1]
        layer_embeddings.append(layer_embedding)
        
    # 存储每个单词的嵌入
    word_embeddings = []
    for text, (chunk_index, start_index, end_index) in zip(tokenized_text, tokenid_pos):
        # 计算指定范围内的平均嵌入，并添加到结果中
        word_embeddings.append(np.average(layer_embeddings[chunk_index][start_index: end_index], axis=0))
        
    word_embeddings = np.array(word_embeddings)
    return word_embeddings


def main():
    # --------- 数据处理 ---------#
    data_folder = 'agnews/dataset'
    os.makedirs(data_folder, exist_ok=True)
    # 检查是否存在 dataset.pk 文件
    pk_file = os.path.join(data_folder, "dataset.pk")
    if os.path.exists(pk_file):
        print("------------ Load existing data ------------")
        with open(pk_file, "rb") as f:
            dataset = pk.load(f)
    else:
        print("------------ Read new data ------------")
        dataset = load('agnews')
        with open(pk_file, "wb") as f:
            pk.dump(dataset, f)
    data = dataset["cleaned_text"]
    data = [x.lower() for x in data]
    
    # --------- 加载模型 ---------#
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model.eval()  
    model.cuda(0)  
    
    #--------- 分词 ---------#
    tokenization_info = []
    counts = Counter()  
    for text in tqdm(data):
        # 返回的 tokenized_text是分词后列表
        # chunk_list 长文本分成的多个子块
        # tokenid_pos 每个token在chunk_list对应的位置 (chunk_id, start_pos, end_pos)
        tokenized_text, tokenid_pos, chunk_list = prepare_tokenized_chunks(tokenizer, text)

        counts.update(word.translate(str.maketrans('','',string.punctuation)) for word in tokenized_text) # 统计词频，移除标点符号
    del counts['']      # 删除空字符串的计数
    updated_counts = {k: c for k, c in counts.items() if c >= vocab_min_occurrence} # 取出词频阈值的词
    
    word_rep = {}  # 存储词的上下文表示
    word_count = {}  # 存储词的计数

    # 计算每个词的上下文表示
    for text in tqdm(data):
        tokenized_text, tokenid_pos, chunk_list = prepare_tokenized_chunks(tokenizer, text)
        tokenization_info.append((tokenized_text, tokenid_pos, chunk_list))
        
        # 获取每个token的上下文词表示
        contextualized_word_representations = process_embeddings(model, layer, tokenized_text,
                                         tokenid_pos, chunk_list)
        
        for i in range(len(tokenized_text)):
            word = tokenized_text[i]
            if word in updated_counts.keys():
                if word not in word_rep:
                    word_rep[word] = 0
                    word_count[word] = 0
                word_rep[word] += contextualized_word_representations[i]
                word_count[word] += 1  # 计数
    
    # s_w = \frac{\sum_{D_i,j=w} t_{i,j}}{\sum_{D_i,j=w} 1} \quad (1)
    # 计算每个词的平均表示
    word_avg = {}
    for k,v in word_rep.items():
        word_avg[k] = word_rep[k]/word_count[k]
    
    vocab_words = list(word_avg.keys()) # 词汇表
    static_word_representations = list(word_avg.values()) # 静态表示
    vocab_occurrence = list(word_count.values()) # 出现次数

    # 保存分词信息
    with open(os.path.join(data_folder, f"tokenization.pk"), "wb") as f:
        pk.dump({
            "tokenization_info": tokenization_info,
        }, f, protocol=4)

    # 保存静态词表示
    with open(os.path.join(data_folder, f"static_repr.pk"), "wb") as f:
        pk.dump({
            "static_word_representations": static_word_representations,
            "vocab_words": vocab_words,
            "word_to_index": {v: k for k, v in enumerate(vocab_words)},
            "vocab_occurrence": vocab_occurrence,
        }, f, protocol=4)

if __name__ == '__main__':
    main()
