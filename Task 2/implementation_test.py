from solution import *
import numpy as np
import torch

data = np.load("./train_data.npz")

x_train = torch.from_numpy(data["train_x"]).reshape([-1, 784])
y_train = torch.from_numpy(data["train_y"])
dataset_train = torch.utils.data.TensorDataset(x_train, y_train)

run_solution(dataset_train=dataset_train)
