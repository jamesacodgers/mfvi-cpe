# This script estimates the optimal observation noise by type II ML
# in the normal-normal linear model for various UCI data sets

import os
import torch
import math
import matplotlib.pyplot as plt

from src.linear_utils import opt_sigma
from src.UCI_data import load_dataset


dtype = torch.float64

P_PRSCISN = 10  



if __name__ == "__main__":

        X_df, y_df = load_dataset("naval")

        X = torch.tensor(X_df.values, dtype=dtype)
        y = torch.tensor(y_df.values.squeeze(), dtype=dtype)


        sigma_hat, _ = opt_sigma(X, y, prior_precision=P_PRSCISN, init_sigma=.1, verbose=True)

        print(sigma_hat)


