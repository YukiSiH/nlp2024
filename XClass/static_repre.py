import os
import string
from tqdm import tqdm
from collections import Counter
import pickle as pk
from datasets import load
from utils import load
from transformers import BertModel, BertTokenizer

# 暂时写死
vocab_min_occurrence = 5
layer = 12

def prepare_tokenized_chunks(tokenizer, text):
    max_model_tokens = 512 
    max_tokens = max_model_tokens - 2  # 需要包含 [CLS]和[SEP]
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
    model.cuda(2)  
    
    #--------- 分词 ---------#
    counts = Counter()  
    for text in tqdm(data):
        # 返回的 tokenized_text是分词后列表
        # chunk_list 长文本分成的多个子块
        # tokenid_pos 每个token在chunk_list对应的位置 (chunk_id, start_pos, end_pos)
        tokenized_text, tokenid_pos, chunk_list = prepare_tokenized_chunks(tokenizer, text)

        counts.update(word.translate(str.maketrans('','',string.punctuation)) for word in tokenized_text) # 统计词频，移除标点符号
    del counts['']      # 删除空字符串的计数
    updated_counts = {k: c for k, c in counts.items() if c >= vocab_min_occurrence} # 取出词频阈值的词

if __name__ == '__main__':
    main()
