import os
import pickle as pk
import numpy as np
from utils import cosine_sim_embeddings

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
        else:
            if word in existing_class_words: # 不是本类，但参与了本类。新的类表示无法使前 $ i $ 个关键词保持不变
                stop_criterion = True
                break
    return highest_sim_word_index, stop_criterion


def update_class_words(cls, vocab_words, highest_sim_word_index, class_words, class_words_representations, static_word_representations, masked_words):
    # 更新类的词集合和表示
    # 加入最大相似度的词
    word = vocab_words[highest_sim_word_index]
    class_words[cls].append(word)
    class_words_representations[cls].append(static_word_representations[highest_sim_word_index])
    masked_words.add(word)


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
            highest_sim_word_index, stop_criterion = \
                find_best_word_for_class(cls, vocab_words, sim, nearest_class, class_words, masked_words)

            if stop_criterion:  # 检查停止条件
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

if __name__ == '__main__':
    main()
