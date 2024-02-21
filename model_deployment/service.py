import numpy as np
import pandas as pd
from pandas import DataFrame

import pickle 

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torch import Tensor

import bentoml

BENTO_AUTOENCODER_TAG = "autoencoder_artifacts:latest"
BENTO_KMEANS_TAG = "kmeans_artifacts:3vmug2gpfs6qokg7"

with open("..\\notebook\\x_train_alpha(1).pkl", 'rb') as file: 
    X_train = pickle.load(file) 

# Loading model from bentoml, and turn it into original model when training
bentoml_autoencoder_model = bentoml.mlflow.load_model(BENTO_AUTOENCODER_TAG)
autoencoder = bentoml_autoencoder_model._model_impl.pytorch_model
bentoml_kmeans_model = bentoml.mlflow.load_model(BENTO_KMEANS_TAG)
kmeans = bentoml_kmeans_model._model_impl.sklearn_model

# Set up device with gpu 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Prepare data
train_by_user = pd.read_csv('C:\\Users\\Theba\\OneDrive\Documents\\MLOps\\realtime-recsys-ops\\dataset\\train_total_by_user_38k1.csv')
test_by_user  = pd.read_csv('C:\\Users\\Theba\\OneDrive\Documents\\MLOps\\realtime-recsys-ops\\dataset\\test_by_user_695.csv')

train_data = train_by_user.loc[:, ['UserID', 'HotelID', 'Rating']]
train_data = train_data.rename(columns={'UserID': 'UID', 'HotelID': 'MID', 'Rating': 'rate'})
test_data = test_by_user.loc[:, ['UserID', 'HotelID', 'Rating']]
test_data = test_data.rename(columns={'UserID': 'UID', 'HotelID': 'MID', 'Rating': 'rate'})

df_user = pd.DataFrame(range(1, len(train_data['UID'].unique())+1), columns=['UID'])

# Functions
def compute_item_ratings(num_clusters: int, train_data: DataFrame) -> np.ndarray:
    """Compute rating of item based on arverage rating in each cluster.

    Args:
        num_clusters (int): Number of cluster.
        train_data (DataFrame)

    Returns:
        np.ndarray: With ``shape=(num_cluster, num_item)``.
    """    
    # Compute item ratings based on cluster similarity.
    num_item = len(train_data['MID'].unique())
    item_ratings = np.zeros((num_clusters, num_item))
    new_index = np.arange(1, num_item + 1)
    for i, cluster_label in enumerate(range(num_clusters)):
        cluster_members = train_data[train_data['cluster_label'] == cluster_label]
        cluster_ratings = cluster_members.groupby('MID')['rate'].mean()
        item_ratings[i, :] = cluster_ratings.reindex(new_index, fill_value=0).reset_index()['rate']

    return item_ratings

def get_top_similar_items(itemid: int, cosine_sim: np.ndarray, topk: int= 10) -> list:
    """Get top `Similarity Items` based cosine sim. 

    Args:
        itemid (int): Item's id we want to find it's top similarity.
        cosine_sim (np.ndarray): Array cotain score similarity between each item.
        topk (int, optional): Top k items similarity with itemid.

    Returns:
        list.
    """    
    sim_scores = list(enumerate(cosine_sim[itemid]))
    sim_item = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    list_sim_itemid = [item[0] for item in sim_item[1:topk]]
    return list_sim_itemid 

def update_rating(item_ratings_matrix: np.ndarray, cosine_sim: np.ndarray, topk: int= 10) -> np.ndarray:
    """There for in a cluster, `there is a item don't be used in this cluster, it's not avarage rating`,
    there for we using this function to create it's rating based on similarity with topk another items.

    Args:
        item_ratings_matrix (np.ndarray): Matrix with `shape(num_cluster, num_item)`.
        cosine_sim (np.ndarray): Array cotain score similarity between each item.

                topk (int, optional): Top k items similarity with itemid. Defaults to 10.

    Returns:
        np.ndarray: Full item_rating_matrix.
    """    
    for i in range(item_ratings_matrix.shape[0]): #tính theo từng cluster
        for item in range(item_ratings_matrix.shape[1]): #tính theo từng item
            if item_ratings_matrix[i][item] == 0: 
                top_sim_item = get_top_similar_items(item, cosine_sim, topk)
                all_score_cluster = [item_ratings_matrix[i][j] for j in top_sim_item if item_ratings_matrix[i][j] !=0]
                avg_rating = sum(all_score_cluster)/(len(all_score_cluster)+1e-5)
                #cập nhật rating cho vị trí bằng 0
                item_ratings_matrix[i][item] = avg_rating
                  
    return item_ratings_matrix

def user_item_matrix(user_cluster_matrix, update_rating_matrix):
    result_dot = np.dot(user_cluster_matrix, update_rating_matrix)
    return result_dot

def get_top_reccommend(userid: int, topk: int, user_rarting_matrix: np.ndarray) -> list:
    """Get `top k items` from user item rate matrix 

    Args:
        userid (int): user id 
        topk (int): topk items to recommend
        user_rarting_matrix (np.ndarray): shape(num_user, num_item)

    Returns:
        list: list of items to recommend
    """    
    list_items_score = list(enumerate(user_rarting_matrix[userid-1]))
    list_items = sorted(list_items_score, key=lambda x: x[1], reverse=True)
    top_list_items = [item[0] for item in list_items[:topk]]
    return top_list_items
    
@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class recommendation:
    @bentoml.api
    def predict(userid: int) -> list: 
        X = X_train.to_numpy()

        # Normalize input features
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_normalized, dtype=torch.float32).to(device)

        # Autoencoder model
        encoded_features = autoencoder(X_train_tensor).detach().numpy().astype('float')
        
        # Kmeans clustering user by encoded features
        cluster_labels = kmeans.predict(encoded_features)
        train_data.loc[:, 'cluster_label'] = cluster_labels[train_data['UID']-1]

        # Create Item rating based on Users in each cluster.
        item_ratings_matrix = compute_item_ratings(num_clusters=len(np.unique(cluster_labels)), train_data=train_data)

        # Classify User into appropriate cluster.
        user_matrix = pd.DataFrame(cluster_labels, columns=['cluster_lable']).reset_index()
        user_cluster_matrix = pd.get_dummies(user_matrix['cluster_lable']).astype(int).to_numpy()

        # Load pre-train score similarity between each item.
        with open('C:\\Users\\Theba\\OneDrive\\Documents\\MLOps\\recsys_ops\\RECSYS_MLOPS\\notebook\\cosine_sim_item_linear_kernel.pkl', 'rb') as file:
            cosine_sim = pickle.load(file)
        
        # Update Item rating and matmul with User Cluster to create User Item Rating Matrix.
        update_rating_matrix = update_rating(item_ratings_matrix, cosine_sim, topk=2)
        user_rating_matrix = user_item_matrix(user_cluster_matrix, update_rating_matrix)

        # Matching top k items with userid
        top_list_item = get_top_reccommend(userid=userid, topk=10, user_rarting_matrix=user_rating_matrix)

        return top_list_item
