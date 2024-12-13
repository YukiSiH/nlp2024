import os  
import pickle as pk 
import numpy as np  
from sklearn.decomposition import PCA  
from sklearn.mixture import GaussianMixture  
from utils import cosine_sim_embeddings  

pca = 64

def main():
    save_dict_data = {}   
    do_pca = pca != 0   
    save_dict_data["pca"] = pca   

    data_dir = 'agnews/dataset'

    with open(os.path.join(data_dir, "dataset.pk"), "rb") as f:
        dictionary = pk.load(f)
        class_names = dictionary["class_names"] 
        num_classes = len(class_names)  

    with open(os.path.join(data_dir, f"document_repr.pk"), "rb") as f:
        dictionary = pk.load(f)
        document_representations = dictionary["document_representations"] 
        class_representations = dictionary["class_representations"]  

        repr_prediction = np.argmax(cosine_sim_embeddings(document_representations, class_representations),
                                    axis=1)
        save_dict_data["repr_prediction"] = repr_prediction

    # 如果需要进行PCA降维
    if do_pca:
        _pca = PCA(n_components=pca, random_state=42)  
        document_representations = _pca.fit_transform(document_representations) 
        class_representations = _pca.transform(class_representations)  # 对类别表示进行降维
        print(f"Explained variance: {sum(_pca.explained_variance_ratio_)}")  # 输出解释方差的和

    cosine_similarities = cosine_sim_embeddings(document_representations, class_representations)
    document_class_assignment = np.argmax(cosine_similarities, axis=1) # 获取每个文档对应的类别索引 
    
    # 创建文档-类别分配矩阵，初始化为零矩阵
    document_class_assignment_matrix = np.zeros((document_representations.shape[0], num_classes))
    for i in range(document_representations.shape[0]):
        document_class_assignment_matrix[i][document_class_assignment[i]] = 1.0  # 将相应的类别位置设为1.0

    # 初始化高斯混合模型 (GMM)
    gmm = GaussianMixture(n_components=num_classes, covariance_type='tied',
                          random_state=42, n_init=999, warm_start=True)
    gmm.converged_ = "HACK"  # 强制修改模型的收敛状态以绕过某些限制

    # 使用文档表示和类别分配矩阵进行GMM模型的初始化
    gmm._initialize(document_representations, document_class_assignment_matrix)
    gmm.lower_bound_ = -np.infty  # 设置模型的初始下界
    gmm.fit(document_representations)  # 拟合GMM模型

    # 获取文档所属类别的预测
    documents_to_class = gmm.predict(document_representations)
    # 获取聚类中心（类别的均值向量）
    centers = gmm.means_
    save_dict_data["centers"] = centers  # 将聚类中心保存到字典中

    # 计算每个文档到类别的距离矩阵
    distance = -gmm.predict_proba(document_representations) + 1
    save_dict_data["documents_to_class"] = documents_to_class  # 将文档的类别分配保存到字典中
    save_dict_data["distance"] = distance  # 将距离矩阵保存到字典中

    with open(os.path.join(data_dir, f"data_alignment.pk"), "wb") as f:
        pk.dump(save_dict_data, f)

if __name__ == '__main__':
    main()  