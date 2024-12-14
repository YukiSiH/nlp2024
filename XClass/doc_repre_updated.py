import os
import pickle as pk
import numpy as np
from scipy.special import softmax
from tqdm import tqdm
from transformers import BertModel
from utils import cosine_sim_embeddings
from static_repre import process_embeddings
import torch
layer = 12
T = 100

def load_data(data_folder):
    # 加载数据集、静态词表示和标记化信息
    with open(os.path.join(data_folder, "dataset.pk"), "rb") as f:
        dataset = pk.load(f)
    
    with open(os.path.join(data_folder, f"static_repr.pk"), "rb") as f:
        vocab = pk.load(f)

    with open(os.path.join(data_folder, f"tokenization.pk"), "rb") as f:
        tokenization_info = pk.load(f)["tokenization_info"]

    return dataset["class_names"], vocab, tokenization_info

def calc_class_representations(vectors):
    # 计算调和加权平均值，第k个关键词的权重为1/k
    num_vectors = len(vectors)
    harmonic_weights = 1.0 / np.arange(1, num_vectors + 1)
    return np.average(vectors, weights=harmonic_weights, axis=0)

# 参数：当前类、词汇表、每个词的最大相似度、最接近的类别、每个类的词集合、已构建的词
def find_best_word_for_class(cls, vocab_words, sim, nearest_class, class_words, masked_words):
    # 找到最适合当前类别的词
    highest_sim = -1.0
    highest_sim_word_index = -1
    lowest_masked_words_sim = 1.0
    existing_class_words = set(class_words[cls]) # 本类里已有的单词
    stop_criterion = False

    for i, word in enumerate(vocab_words):
        if nearest_class[i] == cls: # 如果是本类
            if word not in masked_words: # 没参与构建任何类
                if sim[i] > highest_sim:
                    highest_sim = sim[i]
                    highest_sim_word_index = i
            else:
                if word not in existing_class_words:  # 没参与本类，但参与了其他类；新的类表示无法使前 $ i $ 个关键词保持不变
                    stop_criterion = True
                    break
                lowest_masked_words_sim = min(lowest_masked_words_sim, sim[i]) # 最接近本类，之前也是本类的词，更新本类词汇的最低相似度
        else:
            if word in existing_class_words: # 不是本类，但参与了本类。新的类表示无法使前 $ i $ 个关键词保持不变
                stop_criterion = True
                break
    return highest_sim, highest_sim_word_index, stop_criterion, lowest_masked_words_sim


def update_class_words(cls, vocab_words, highest_sim_word_index, class_words, class_words_representations, static_word_representations, masked_words):
    # 更新类的词集合和表示
    # 加入最大相似度的词
    word = vocab_words[highest_sim_word_index]
    class_words[cls].append(word)
    class_words_representations[cls].append(static_word_representations[highest_sim_word_index])
    masked_words.add(word)

# one-to-one 
def rank_by_significance(embeddings, class_embeddings):
    sim = cosine_sim_embeddings(embeddings, class_embeddings)
    significance_score = [np.max(softmax(similarity)) for similarity in sim] # 对每个输入嵌入计算其与类别嵌入的最大相似度，并通过 softmax 标准化
    significance_ranking = {i: r for r, i in enumerate(np.argsort(-np.array(significance_score)))} # 降序排序并排名
    return significance_ranking # 返回排名

# one-to-all
def rank_by_relation(embeddings, class_embeddings):
    relation_score = cosine_sim_embeddings(embeddings, [np.average(class_embeddings, axis=0)]).reshape((-1)) # 计算输入嵌入与类别嵌入平均值的余弦相似度
    relation_ranking = {i: r for r, i in enumerate(np.argsort(-np.array(relation_score)))}
    return relation_ranking

def mul(l):
    m = 1
    for x in l:
        m *= x + 1
    return m

# 计算排序加权
def weights_from_ranking(rankings):
    if type(rankings[0]) == type(0):
        rankings = [rankings]

    rankings_len = len(rankings[0])
    
    total_score = []
    for i in range(rankings_len):
        total_score.append(mul(ranking[i] for ranking in rankings)) # 每个索引在所有排名中的乘积结果
    
    total_ranking = {i: r for r, i in enumerate(np.argsort(np.array(total_score)))}  # 生成按得分排序后的排名字典，键是原始索引，值是该索引的综合得分在所有索引中的排序结果
    
    weights = [0.0] * rankings_len
    # 计算每个排名位置的权重，权重为排名位置的倒数
    for i in range(rankings_len):
        weights[i] = 1. / (total_ranking[i] + 1)
    
    return weights # 返回计算后的权重列表

def weight_sentence_with_attention(model, vocab, tokenization_info, class_representations, layer):
    # 解包元组
    tokenized_text, tokenid_pos, chunk_list = tokenization_info
    
    # 获取上下文嵌入
    contextualized_word_representations = process_embeddings(
        model, layer, tokenized_text, tokenid_pos, chunk_list
    )
    
    static_representations = []
    contextualized_representations = []
    static_word_representations = vocab["static_word_representations"]
    word_to_index = vocab["word_to_index"]
    
    for i, token in enumerate(tokenized_text):
        if token in word_to_index:
            static_representations.append(static_word_representations[word_to_index[token]])
            contextualized_representations.append(contextualized_word_representations[i])
    
    if len(contextualized_representations) == 0:
        print("Empty sentence")
        return np.average(contextualized_word_representations, axis=0)

    # 获取tokenization_info中的input_ids和attention_mask
    input_ids = torch.tensor([chunk_list[0]], device=model.device)  # 假设你只取第一个块作为input
    attention_mask = torch.ones(input_ids.shape, device=model.device)  # 假设没有pad，全部mask为1

    # 传递给模型并获取attention权重
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)  # 获取模型输出
    attention_weights = outputs[-1]  # 获取最后一层的自注意力权重

    # 计算每个单词的自注意力权重
    attention_score = np.mean(attention_weights[0].cpu().detach().numpy(), axis=1)  # 选择第一个头部并平均

    # 初始化total_weights，假设你用rank或者其他方法来计算它们
    total_weights = np.ones(len(contextualized_representations))  # 默认给每个token权重为1
    
    # 使用当前的相似度权重计算
    significance_ranking = rank_by_significance(contextualized_representations, class_representations)
    relation_ranking = rank_by_relation(contextualized_representations, class_representations)
    significance_ranking_static = rank_by_significance(static_representations, class_representations)
    relation_ranking_static = rank_by_relation(static_representations, class_representations)

    # 使用自注意力权重与其他权重融合
    attention_weights_avg = np.mean(attention_score, axis=1)  # 平均所有单词的注意力权重
    attention_weights_avg = attention_weights_avg.flatten()  # 确保是1D向量

    # 确保total_weights和attention_weights_avg形状匹配
    if len(total_weights) != len(attention_weights_avg):
        print(f"Warning: total_weights length {len(total_weights)} and attention_weights_avg length {len(attention_weights_avg)} do not match.")
        # 处理形状不匹配的情况
        min_len = min(len(total_weights), len(attention_weights_avg))
        total_weights = total_weights[:min_len]
        attention_weights_avg = attention_weights_avg[:min_len]

    # 合并权重
    combined_weights = total_weights * (1 + attention_weights_avg)  # 例如通过加权方式融合
    combined_weights /= np.sum(combined_weights)  # 归一化

    # 使用加权平均计算文档表示
    document_representation = np.average(contextualized_representations, weights=combined_weights, axis=0)
    return document_representation




def weight_sentence(model, vocab, tokenization_info, class_representations, layer):
    tokenized_text, tokenid_pos, tokenids_chunks = tokenization_info
    contextualized_word_representations = process_embeddings(
        model, layer, tokenized_text, tokenid_pos, tokenids_chunks
    )

    # delete
    if len(tokenized_text) != len(contextualized_word_representations):
        raise ValueError("Not match")

    static_representations = []
    contextualized_representations = []
    static_word_representations = vocab["static_word_representations"]
    word_to_index = vocab["word_to_index"]
    for i, token in enumerate(tokenized_text):
        if token in word_to_index:
            static_representations.append(static_word_representations[word_to_index[token]])
            contextualized_representations.append(contextualized_word_representations[i])
    
    if len(contextualized_representations) == 0:  # 上下文化表示为空
        print("Empty sentence")
        return np.average(contextualized_word_representations, axis=0)
    
    # 上下文化和静态表示，one-to-one/one-to-all
    significance_ranking = rank_by_significance(contextualized_representations, class_representations)
    relation_ranking = rank_by_relation(contextualized_representations, class_representations)
    significance_ranking_static = rank_by_significance(static_representations, class_representations)
    relation_ranking_static = rank_by_relation(static_representations, class_representations)
    
    # 根据指定的注意力机制选择权重
    weights = weights_from_ranking((significance_ranking, relation_ranking, significance_ranking_static, relation_ranking_static))

    # 使用加权平均计算文档表示
    document_representation = np.average(contextualized_representations, weights=weights, axis=0)
    return document_representation

def main():
    data_folder = 'agnews/dataset'

    # ---------- 加载数据 --------- #
    class_names, vocab, tokenization_info = load_data(data_folder)
    static_word_representations = vocab["static_word_representations"]
    word_to_index = vocab["word_to_index"]
    vocab_words = vocab["vocab_words"]
    print("Finish reading")
    print(class_names)

    finished_class = set()  # 已完成处理的类别集合
    masked_words = set(class_names)  # 已用于构建类别集合的词，避免在后续处理中重复使用这些词，最初是类别名称集合
    cls_repr = [None for _ in range(len(class_names))]  # 存储每个类别的表示，初始化为None
    class_words = [[class_name] for class_name in class_names]  # 每个类别的初始词集合，仅包含其类别名称
    # 迭代地找到每个类别的下一个关键词，并通过对所有找到的关键词进行加权平均来重新计算类别表示
    # 对于给定的类别，列表中的第一个关键词总是类名
    class_words_representations = [[static_word_representations[word_to_index[class_name]]]
                                   for class_name in class_names]  # 每个类别的初始词表示集合，初始是类别名称的静态词表示

    # 最多迭代 T 次
    # 第 i 次迭代中，我们检索与当前类表示最相似的列表外单词，并基于前 i+1 个单词计算新的类表示。
    for t in range(1, T):
        # 计算每个类别的平均表示，调和加权
        class_representations = [calc_class_representations(rep) for rep in class_words_representations]

        # 计算每个词与类别表示的余弦相似度
        cosine_sim = cosine_sim_embeddings(static_word_representations, class_representations)
        nearest_class = cosine_sim.argmax(axis=1)  # 每个词最接近的类别索引
        sim = cosine_sim.max(axis=1)  # 每个词的最大相似度

        for cls in range(len(class_names)):
            if cls in finished_class:  # 如果该类别已完成处理，跳过
                continue

            # 寻找当前类别的最佳词，传入最接近的类别索引、最大相似度、词汇表、每个类的词集合、已构建的词、当前类
            highest_sim, highest_sim_word_index, stop_criterion, lowest_masked_words_sim = \
                find_best_word_for_class(cls, vocab_words, sim, nearest_class, class_words, masked_words)

            # 新的类表示下，原有的词汇不能作为类表示了，导致集合改变，此时停止
            if lowest_masked_words_sim < highest_sim or stop_criterion:  # 检查停止条件
                finished_class.add(cls)  # 将类别标记为完成
                class_words[cls].pop()  # 移除最后一个词
                class_words_representations[cls].pop()  # 移除最后一个词表示
                cls_repr[cls] = calc_class_representations(class_words_representations[cls])  # 更新类别表示
                print(class_words[cls])  # 打印当前类别的词集合
                break

            # 更新类别词集合和表示
            update_class_words(cls, vocab_words, highest_sim_word_index, class_words, class_words_representations, static_word_representations, masked_words)
            cls_repr[cls] = calc_class_representations(class_words_representations[cls])  # 更新类别表示

        if len(finished_class) == len(class_names):  # 检查是否所有类别都已完成处理
            break

    class_representations = np.array(cls_repr)

    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model.eval()  
    model.cuda(0)  

    # 计算文档表示
    # 遵循注意力机制，根据单词与类别表示的相似性来为单词分配权重
    # 为每段text生成一个加权表示
    document_representations = [
        weight_sentence_with_attention(model, vocab, _tokenization_info, class_representations,  layer)
        for _tokenization_info in tqdm(tokenization_info, total=len(tokenization_info))
    ]

    document_representations = np.array(document_representations)
    print("Finish reading document representations")

    with open(os.path.join(data_folder, f"document_repr_updated.pk"), "wb") as f:
        pk.dump({
            "class_words": class_words,  # 每个类别的词集合
            "class_representations": class_representations,  # 类别表示
            "document_representations": document_representations,  # 文档表示
        }, f, protocol=4)


if __name__ == '__main__':
    main()
