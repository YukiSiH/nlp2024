import os
import pickle
from shutil import copyfile
from utils import clean_str, evaluate_predictions
import numpy as np

# 动态调整的全局参数
confidence_percentile = 50  # 置信度阈值的分位数

def write_to_dir(text, labels, dir_name):
    os.makedirs(os.path.join('agnews', dir_name), exist_ok=True)

    with open(os.path.join('agnews', dir_name, "dataset.txt"), "w") as f:
        for line in text:
            f.write(line)
            f.write("\n")

    with open(os.path.join('agnews', dir_name, "labels.txt"), "w") as f:
        for label in labels:
            f.write(str(label))
            f.write("\n")

    copyfile(os.path.join('agnews', "classes.txt"),
             os.path.join('agnews', dir_name, "classes.txt"))

def calculate_dynamic_threshold(distances, percentile):
    """
    根据指定的分位数计算每个类别的动态置信度阈值。
    :param distances: 每个类别的距离列表
    :param percentile: 置信度阈值的分位数（0-100）
    :return: 每个类别的动态阈值
    """
    return [np.percentile(dist, percentile) for dist in distances]

def main():
    with open('agnews/dataset.txt', mode='r', encoding='utf-8') as text_file:
        text = list(map(lambda x: x.strip(), text_file.readlines()))
    cleaned_text = [clean_str(doc) for doc in text]

    # 加载已保存的数据
    with open('agnews/dataset/data_alignment_updated.pk', "rb") as f:
        save_data = pickle.load(f)
        documents_to_class = save_data["documents_to_class"]  # 文档对应的分类
        distance = save_data["distance"]  # 距离矩阵
        num_classes = distance.shape[1]  # 类别数量

    pseudo_document_class_with_confidence = [[] for _ in range(num_classes)]
    for i in range(documents_to_class.shape[0]):
        pseudo_document_class_with_confidence[documents_to_class[i]].append((distance[i][documents_to_class[i]], i))

    # 计算动态阈值
    class_distances = [
        [x[0] for x in pseudo_document_class_with_confidence[i]]
        for i in range(num_classes)
    ]
    dynamic_thresholds = calculate_dynamic_threshold(class_distances, confidence_percentile)

    selected = []
    not_selected = list(range(len(cleaned_text)))  # 初始化未选中的文档列表
    gold_labels = list(map(int, open('agnews/labels.txt').readlines()))

    # 按动态阈值选择高置信度的文档
    for i in range(num_classes):
        pseudo_document_class_with_confidence[i] = sorted(pseudo_document_class_with_confidence[i])
        confident_documents = [x for x in pseudo_document_class_with_confidence[i] if x[0] <= dynamic_thresholds[i]]
        confident_documents = [x[1] for x in confident_documents]
        selected.extend(confident_documents)

    # 过滤未选中的文档
    selected = sorted(selected)
    not_selected = sorted(set(not_selected) - set(selected))

    # 获取选中和未选中文档的文本和类别
    selected_text = [cleaned_text[i] for i in selected]
    selected_classes = [documents_to_class[i] for i in selected]

    not_selected_text = [cleaned_text[i] for i in not_selected]
    not_selected_classes = [documents_to_class[i] for i in not_selected]

    # 获取选中文档的实际类别
    gold_classes = [gold_labels[i] for i in selected]
    evaluate_predictions(gold_classes, selected_classes)  # 评估预测结果

    write_to_dir(selected_text, selected_classes, "agnews_train_updated")  # 写入选中的数据
    write_to_dir(not_selected_text, not_selected_classes, "agnews_eval_updated")  # 写入未选中的数据

if __name__ == '__main__':
    main()
