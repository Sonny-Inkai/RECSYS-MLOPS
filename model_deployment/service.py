import os
import json

import numpy as np
import pandas as pd
from pandas import DataFrame

import math
import pickle 

import networkx as nx
import itertools
import collections

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torch import Tensor

import time

import ray 
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler

import mlflow
from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow

import bentoml

BENTO_AUTOENCODER_TAG = "autoencoder_artifacts:latest"
BENTO_KMEANS_TAG = "kmeans_artifacts:3vmug2gpfs6qokg7"

with open("..\\notebook\\x_train_alpha(1).pkl", 'rb') as file: 
    X_train = pickle.load(file) 

X_train2 = X_train
X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)


autoencoder = bentoml.mlflow.get(BENTO_AUTOENCODER_TAG).to_runner()
kmeans = bentoml.mlflow.get(BENTO_KMEANS_TAG).to_runner()

svc = bentoml.Service('ghrs', runners=[autoencoder, kmeans])

@bentoml.service()
def predict(userid: int) -> list: 
    encoded_featuers = autoencoder.predict.run(X_train)
    cluster_labels = kmeans.predict.run(encoded_featuers)
    #print(cluster_labels)
    return cluster_labels


predict