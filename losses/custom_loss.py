import torch.nn.functional as F
import torch
import math
from utils import check_nan_inf


def bivariate_normal(x1, x2, mu1, mu2, sigma1, sigma2, corr):
    # x1, x2 are [batch, seq_len, 1],
    # mu1, mu2 are [batch, seq_len, K],
    # sigma1, sigma2, corr are [batch, seq_len, K]
    # print(x1.shape, x2.shape, mu1.shape, mu2.shape,
    #       sigma1.shape, sigma2.shape, corr.shape)
    # equation 25
    Z = ((x1 - mu1) ** 2) / (sigma1 ** 2) + ((x2 - mu2) ** 2) / \
        (sigma2 ** 2) - 2 * corr * (x1 - mu1) * (x2 - mu2) / (sigma1 * sigma2)
    # Z is [batch, seq_len, K]
    # check_nan_inf(Z, "Z")
    # equation 24
    norm = 2 * math.pi * sigma1 * sigma2 * \
        torch.sqrt(1 - corr ** 2)  # [batch, seq_len, K]
    # check_nan_inf(norm, "norm")
    # equation 23
    res = torch.exp(-Z / (2 * (1 - corr ** 2))) / norm  # [batch, seq_len, K]
    # check_nan_inf(res, "res")

    return res


def MDN_loss_function(output, y):
    # y is [batch, seq_len, 3]
    # pi is [batch, seq_len, K],
    # mu1, mu2 are [batch, seq_len, K],
    # sigma1, sigma2 are [batch, seq_len, K],
    # corr is [batch, seq_len, K],
    # eot is [batch, seq_len, 1]
    pi, mu1, mu2, sigma1, sigma2, corr, eot = output

    # Compute loss
    x1, x2 = y[:, :, 1], y[:, :, 2]  # [batch, seq_len]

    # Get bivariate normal distribution
    # prob_x1_x2 is [batch, seq_len, K]
    prob_x1_x2 = pi * \
        bivariate_normal(x1.unsqueeze(-1), x2.unsqueeze(-1),
                         mu1, mu2, sigma1, sigma2, corr)
    # prob_x1_x2 is [batch, seq_len]
    prob_x1_x2 = torch.sum(prob_x1_x2, dim=-1)  # Sum over components
    # check_nan_inf(prob_x1_x2, "prob_x1_x2")

    # Calculate the end of stroke loss
    x3 = y[:, :, 0].unsqueeze(-1)
    # equation 26
    prob_x3 = (eot * x3 + (1 - eot) * (1 - x3)).squeeze(-1)  # [batch, seq_len]
    if not check_nan_inf(prob_x3, "prob_x3"):
        # print("x3", x3)
        # print("eot", eot)
        pass

    # Compute negative log likelihood for each point in the sequence
    # equation 26
    # Added epsilon to prevent log(0)
    loss = -torch.log(prob_x1_x2 + 1e-5) - \
        torch.log(prob_x3 + 1e-5)  # range (0, inf), shape [batch, seq_len]

    # check_nan_inf(loss, "loss")
    # print("prob_x1_x2", prob_x1_x2)
    # print("prob_x3", prob_x3)

    return torch.mean(loss)  # Return mean loss over the sequence
