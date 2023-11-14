import jax.numpy as jnp
from jax import vmap
import jax
from utils import check_nan_inf
import numpy as np


def bivariate_normal(x1, x2, mu1, mu2, sigma1, sigma2, corr):
    """
    Computes the PDF of a bivariate normal distribution with parameters mu1, mu2,
    sigma1, sigma2, and correlation corr for the points in (x1, x2).
    """
    Z = jnp.square((x1 - mu1) / sigma1) + jnp.square((x2 - mu2) / sigma2) - \
        2 * corr * (x1 - mu1) * (x2 - mu2) / (sigma1 * sigma2)
    norm = 2 * jnp.pi * sigma1 * sigma2 * jnp.sqrt(1 - jnp.square(corr))
    return jnp.exp(-Z / (2 * (1 - jnp.square(corr)))) / norm


def mdn_loss_function(output, y):
    """
    Computes the MDN loss function for a batch of predictions and targets.
    """
    # output is [batch, seq_len, 6 * k + 1]
    params = jnp.split(
        output[:, :, :-1], 6, axis=-1)
    pi_raw, mu1, mu2, sigma1_raw, sigma2_raw, corr_raw = params
    pi = jax.nn.softmax(pi_raw, axis=-1)
    sigma1 = jnp.exp(sigma1_raw)
    sigma2 = jnp.exp(sigma2_raw)
    corr = jnp.tanh(corr_raw)
    eos_raw = output[:, :, -1][:, :, None]
    eos = jax.nn.sigmoid(eos_raw)
    x1, x2, x3 = y[:, :, 1], y[:, :, 2], y[:, :, 0]

    prob_x1_x2 = pi * vmap(bivariate_normal, in_axes=(None, None, 0, 0, 0, 0, 0))(
        x1[:, :, None], x2[:, :, None], mu1, mu2, sigma1, sigma2, corr)
    prob_x1_x2 = jnp.sum(prob_x1_x2, axis=-1)

    prob_x3 = (eos * x3[:, :, None] + (1 - eos)
               * (1 - x3[:, :, None])).squeeze(-1)

    loss = -jnp.log(prob_x1_x2 + 1e-8) - jnp.log(prob_x3 + 1e-8)

    return jnp.mean(loss)
