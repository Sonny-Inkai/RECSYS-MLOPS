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

X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)


autoencoder = bentoml.mlflow.get(tag_like=BENTO_AUTOENCODER_TAG).to_runner()
autoencoder.init_local()
svc = bentoml.Service('ghrs', runners=[autoencoder])

@svc.api(input=, output=torch.Tensor)
def predict(input: Tensor) -> Tensor: 
    a = autoencoder.predict.run(input)
    print(a)

if __name__ == "__main__":
    predict(X_train_tensor)

