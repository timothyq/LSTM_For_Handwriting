import torch
import torch.nn as nn
from utils import check_nan_inf
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from torch.distributions import Categorical, MultivariateNormal


class MixtureDensityLayer(nn.Module):
    def __init__(self, num_mixtures):
        super(MixtureDensityLayer, self).__init__()
        self.num_mixtures = num_mixtures

    def forward(self, output):
        # output is [batch, seq_len, 6*K+1]
        K = self.num_mixtures
        pi_logits, mu1, mu2, std_exp1, std_exp2, corr_tanh, eos = torch.split(
            output, K, dim=-1)
        # Assuming eot_logit is a single value, not a vector like the others.
        # eot_logit = output[:, :, -1]

        pi = F.softmax(pi_logits, dim=-1)
        std1 = torch.exp(std_exp1)
        std2 = torch.exp(std_exp2)
        corr = torch.tanh(corr_tanh)
        eos_p = torch.sigmoid(eos)

        # check_nan_inf(pi, "pi")
        # check_nan_inf(mu1, "mu1", check_neg=False)
        # check_nan_inf(mu2, "mu2", check_neg=False)
        # check_nan_inf(std1, "std1")
        # check_nan_inf(std2, "std2")
        # check_nan_inf(corr, "corr", check_neg=False)
        # check_nan_inf(eos_p, "eos_p", upper_bound=1.0, lower_bound=0.0)

        # Reshape tensors
        # print("output", output.shape)
        # print("mu1", mu1.shape)
        # print("std1", std1.shape)
        # [batch, seq_len, K, 2]

        return pi, mu1, mu2, std1, std2, corr, eos_p  # [batch, seq_len, K]


class Uncondition_LSTM(nn.Module):
    def __init__(self, alphabet_size, input_size, hidden_size, num_layers, component_K, dropout=0):
        super(Uncondition_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = alphabet_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = component_K * 6 + 1
        self.dropout = dropout

        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True, dropout=dropout)

        self.layer_norms = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, self.output_size)
        self.mdn = MixtureDensityLayer(component_K)

    def forward(self, x):
        # x is [batch, seq_len, input_size]
        x, _ = self.lstm(x)
        x = self.layer_norms(x)
        output = self.fc(x)  # [batch, seq_len, output_size]
        output = self.mdn(output)  # 7 * [batch, seq_len, K]

        return output  # 7 * [batch, seq_len, K]

    def predict(self, sequence_length, device):
        # [batch=1, seq_len=1, input_size]
        current_input = torch.zeros(1, 1, 3, device=device)
        generated_sequence = current_input

        for _ in range(sequence_length):
            # Generate output for the current input
            output = self.forward(current_input)  # 7 * [batch=1, seq_len=1, K]

            # Sample from the output distribution (MDN parameters)
            next_point = sample_from_mdn(output)  # [batch=1, seq_len=1, 3]

            # Append the generated point to the sequence
            generated_sequence = torch.cat(
                (generated_sequence, next_point), dim=1)  # [batch=1, seq_len, 3]

            # Prepare the input for the next step
            current_input = next_point

            # Optionally, check for end of stroke and break if needed

        return generated_sequence

    # might also need a function to initialize the hidden state, especially if we're
    # not using batches. The paper seems to use online learning which doesn't use batches.

    def _initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # initializing biases for LSTM forget gate to 1?
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.)

        init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)


class Condition_LSTM2(nn.Module):
    def __init__(self, alphabet_size, window_K, input_size, hidden_size, num_layers, component_K, dropout=0):
        super(Condition_LSTM2, self).__init__()

        self.input_size = input_size  # shape of input is [batch, input_size]
        self.hidden_size = hidden_size
        self.alphabet_size = alphabet_size
        self.window_K = window_K
        self.num_layers = num_layers
        self.output_size = component_K * 6 + 1
        self.dropout = dropout

        # LSTM layers
        self.lstm1 = nn.LSTM(input_size + alphabet_size, hidden_size,
                             num_layers=num_layers, batch_first=True, dropout=dropout)
        # self.fc1 = nn.Linear(hidden_size, 3 * window_K)
        self.lstm2 = nn.LSTM(
            input_size + alphabet_size + hidden_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)

        # output is [batch, 3 * window_K]
        self.alpha_beta_kappa = nn.Linear(hidden_size, 3 * window_K)

        self.fc2 = nn.Linear(hidden_size, self.output_size)

    def compute_phi(self, alpha, beta, kappa, sequence_c):
        # sequence_c is [batch, seq_len_c, alphabet_size]
        # alpha, beta, kappa are [batch, window_K]
        u = torch.arange(sequence_c.size(1)).float().to(
            sequence_c.device).unsqueeze(0).unsqueeze(2)  # [1, seq_len_c, 1]

        phi = torch.sum(alpha.unsqueeze(1) * torch.exp(-beta.unsqueeze(1) *
                        (kappa.unsqueeze(1) - u) ** 2), dim=2)  # phi is [batch, seq_len_c]
        w = torch.bmm(phi.unsqueeze(1), sequence_c).squeeze(
            1)  # w is [batch, alphabet_size]
        return w

    def split_and_transform_output(output):
        # output is [batch, seq_len, 6*K+1]
        K = output.shape[-1] // 6  # Deduce the value of K from output shape

        # Split the output into GMM parameters
        pi_logits = output[:, :, :K]  # [batch, seq_len, K]
        mu1 = output[:, :, K:2*K]
        mu2 = output[:, :, 2*K:3*K]
        std_exp1 = output[:, :, 3*K:4*K]
        std_exp2 = output[:, :, 4*K:5*K]
        corr_tanh = output[:, :, 5*K:6*K]
        eot_logit = output[:, :, 6*K]

        # Apply activation functions
        # Ensure component weights sum up to 1.
        pi = F.softmax(pi_logits, dim=-1)  # [batch, seq_len, K]
        # Ensure std is positive. [batch, seq_len, K]
        std1 = torch.exp(std_exp1)
        std2 = torch.exp(std_exp2)  # Ensure std is positive.
        corr = torch.tanh(corr_tanh)  # Ensure correlation is between [-1, 1]
        # Ensure eot is between [0, 1]
        eot = torch.sigmoid(eot_logit.unsqueeze(-1))
        # check_nan_inf(pi, "pi")
        # check_nan_inf(mu1, "mu1", check_neg=False)
        # check_nan_inf(mu2, "mu2", check_neg=False)
        # check_nan_inf(std1, "std1")
        # check_nan_inf(std2, "std2")
        # check_nan_inf(corr, "corr", check_neg=False)
        # check_nan_inf(eot, "eot", upper_bound=1.0, lower_bound=0.0)

        # Reshape tensors
        # print("output", output.shape)
        # print("mu1", mu1.shape)
        # print("std1", std1.shape)
        # [batch, seq_len, K, 2]

        return pi, mu1, mu2, std1, std2, corr, eot

    def forward(self, x, c_seq, h1_prev=None, c1_prev=None, h2_prev=None, c2_prev=None, kappa_prev=None, w_t_pre=None):
        # Assuming x is of shape [batch, input_size]
        batch_size, _ = x.size()
        # check_nan_inf(x, "x")

        if h1_prev is None or c1_prev is None or h2_prev is None or c2_prev is None:
            h1_prev = c1_prev = torch.zeros(
                batch_size, self.hidden_size).to(x.device)
            h2_prev = c2_prev = torch.zeros(
                batch_size, self.hidden_size).to(x.device)
        # check_nan_inf(h1_prev, "h1_prev")
        # check_nan_inf(c1_prev, "c1_prev")

        if kappa_prev is None:
            kappa_prev = torch.zeros(batch_size, self.window_K).to(x.device)
        if w_t_pre is None:
            w_t_pre = torch.zeros(batch_size, self.alphabet_size).to(x.device)

        # Forward through lstm1
        # h1_out is [batch, seq_len = 1, hidden_size]
        h1_out, (h1_n, c1_n) = self.lstm1(torch.cat([x.unsqueeze(1), w_t_pre.unsqueeze(
            1)], dim=2), (h1_prev.unsqueeze(0), c1_prev.unsqueeze(0)))
        h1_prev, c1_prev = h1_n.squeeze(
            0), c1_n.squeeze(0)  # [batch, hidden_size]
        # check_nan_inf(h1_out, "h1_out")
        # check_nan_inf(h1_prev, "h1_prev")
        # check_nan_inf(c1_prev, "c1_prev")

        # Calculate attention weights and context vector (w_t) for lstm1
        # alpha, beta, kappa are [batch, window_K]
        abp = self.alpha_beta_kappa(h1_out.squeeze(1))
        alpha_hat, beta_hat, kappa_hat = torch.chunk(
            abp, 3, dim=1)
        alpha = torch.exp(alpha_hat)
        beta = torch.exp(beta_hat)
        kappa = kappa_prev + torch.exp(kappa_hat)
        # check_nan_inf(alpha, "alpha")
        # check_nan_inf(beta, "beta")
        # check_nan_inf(kappa, "kappa")

        # alpha, beta, kappa are [batch, window_K], c_seq is [batch, seq_len_c, alphabet_size]
        # w_t is [batch, alphabet_size]
        w_t = self.compute_phi(alpha, beta, kappa, c_seq)
        # check_nan_inf(w_t, "w_t")

        # Forward through lstm2
        # h2_out is [batch, seq_len = 1, hidden_size]
        h2_out, (h2_n, c2_n) = self.lstm2(torch.cat([x.unsqueeze(1), w_t.unsqueeze(1), h1_out], dim=2),
                                          (h2_prev.unsqueeze(0), c2_prev.unsqueeze(0)))
        h2_prev, c2_prev = h2_n.squeeze(0), c2_n.squeeze(0)
        # check_nan_inf(h2_out, "h2_out")

        w_t_pre = w_t
        # check_nan_inf(w_t_pre, "w_t_pre")

        # Compute output
        out = self.fc2(h2_out.squeeze(1))  # [batch, output_size]
        # outputs.append(out)

        return out, (h1_prev, c1_prev, h2_prev, c2_prev, kappa, w_t_pre)


class Condition_LSTM(nn.Module):
    def __init__(self, alphabet_size, window_K, input_size, hidden_size, num_layers, component_K, dropout=0):
        super(Condition_LSTM, self).__init__()

        self.input_size = input_size  # shape of input is [batch, input_size]
        # print("input_size", input_size)
        self.hidden_size = hidden_size
        # print("hidden_size", hidden_size)
        self.alphabet_size = alphabet_size
        # print("alphabet_size", alphabet_size)
        self.window_K = window_K
        self.num_layers = num_layers
        self.output_size = component_K * 6 + 1
        self.dropout = dropout

        self.lstm1 = nn.LSTM(input_size + alphabet_size, hidden_size,
                             num_layers=num_layers, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(
            input_size + alphabet_size + hidden_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.alpha_beta_kappa = nn.Linear(hidden_size, 3 * window_K)
        self.fc2 = nn.Linear(hidden_size, self.output_size)
        self.mdn = MixtureDensityLayer(component_K)

    def compute_phi(self, alpha, beta, kappa, sequence_c):
        # alpha, beta, kappa are [batch, window_K]
        # sequence_c is [batch, seq_len_c, alphabet_size]
        u = torch.arange(sequence_c.size(1)).float().to(
            sequence_c.device).unsqueeze(0).unsqueeze(2)  # [1, seq_len_c, 1]
        phi = torch.sum(alpha.unsqueeze(1) * torch.exp(-beta.unsqueeze(1) *
                        (kappa.unsqueeze(1) - u) ** 2), dim=2)  # phi is [batch, seq_len_c]
        # print("alpha", alpha.shape)
        # print("phi", phi.shape)
        # print("sequence_c", sequence_c.shape)
        w = torch.bmm(phi.unsqueeze(1), sequence_c).squeeze(1)
        return w

    def forward(self, x, c_seq, h1_prev=None, c1_prev=None, h2_prev=None, c2_prev=None, kappa_prev=None, w_t_pre=None):
        # input shape  [batch, seq_len, input_size]
        # c_seq shape [batch, seq_len, alphabet_size]
        # print("x", x.shape)
        batch_size, seq_len, _ = x.size()

        if h1_prev is None or c1_prev is None or h2_prev is None or c2_prev is None:
            h1_prev = c1_prev = torch.zeros(
                self.num_layers, batch_size, self.hidden_size).to(x.device)
            h2_prev = c2_prev = torch.zeros(
                self.num_layers, batch_size, self.hidden_size).to(x.device)

        if kappa_prev is None:
            kappa_prev = torch.zeros(batch_size, self.window_K).to(x.device)
        if w_t_pre is None:
            w_t_pre = torch.zeros(
                batch_size, self.alphabet_size).to(x.device)

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            # LSTM1
            # x_t is (batch, 1, input_size),
            # print("x_t", x_t.shape)
            h1_out, (h1_prev, c1_prev) = self.lstm1(
                torch.cat([x_t.unsqueeze(1), w_t_pre.unsqueeze(1)], dim=2), (h1_prev, c1_prev))
            # h1_out is [batch, seq_len = 1, hidden_size]

            # Attention mechanism
            # print("h1_out", h1_out.shape)
            abp = self.alpha_beta_kappa(
                h1_out.squeeze(1))  # [batch, 3 * window_K]
            alpha_hat, beta_hat, kappa_hat = torch.chunk(
                abp, 3, dim=1)  # [batch, window_K]
            alpha = torch.exp(alpha_hat)  # [batch, window_K]
            beta = torch.exp(beta_hat)  # [batch, window_K]
            kappa = kappa_prev + torch.exp(kappa_hat)  # [batch, window_K]
            # [batch, alphabet_size]
            w_t = self.compute_phi(alpha, beta, kappa, c_seq)

            # LSTM2
            # print("input", torch.cat([x_t.unsqueeze(1), w_t.unsqueeze(1), h1_out], dim=2).shape)
            # print(self.lstm2)
            # print("x_t", x_t.shape)
            # print("w_t", w_t.shape)
            # print("h1_out", h1_out.shape)
            h2_out, (h2_prev, c2_prev) = self.lstm2(torch.cat(
                [x_t.unsqueeze(1), w_t.unsqueeze(1), h1_out], dim=2), (h2_prev, c2_prev))
            w_t_pre = w_t

            # Output
            out = self.fc2(h2_out.squeeze(1))  # [batch, output_size]
            outputs.append(out)
        outputs = torch.stack(outputs, dim=1)
        # outputs is [batch, seq_len, output_size]
        return self.mdn(outputs)

    def predict(self, initial_input, c_seq, seq_len, threshold=0.6):
        self.eval()
        # Initialize hidden states, cell states, kappa, and w_t_pre
        batch_size = initial_input.size(0)
        # print("c_seq", c_seq.shape)

        # Initialize hidden states and cell states for both LSTM layers
        h1_prev = torch.zeros(self.num_layers, batch_size,
                              self.hidden_size).to(initial_input.device)
        c1_prev = torch.zeros(self.num_layers, batch_size,
                              self.hidden_size).to(initial_input.device)
        h2_prev = torch.zeros(self.num_layers, batch_size,
                              self.hidden_size).to(initial_input.device)
        c2_prev = torch.zeros(self.num_layers, batch_size,
                              self.hidden_size).to(initial_input.device)

        # Initialize kappa_prev and w_t_pre
        kappa_prev = torch.zeros(
            batch_size, self.window_K).to(initial_input.device)
        w_t_pre = torch.zeros(batch_size, self.alphabet_size).to(
            initial_input.device)
        # Shape becomes [batch_size, 1, alphabet_size]
        w_t_pre = w_t_pre.unsqueeze(1)

        # Starting input
        # initial_input should be a tensor with shape [batch, 3] representing (eot, x1, x2)
        x = initial_input

        predictions = []

        for _ in range(seq_len):
            # Forward pass through LSTM1
            # Concatenate x and w_t_pre and unsqueeze to add sequence length dimension
            lstm1_input = torch.cat(
                [x, w_t_pre], dim=2)

            h1_out, (h1_n, c1_n) = self.lstm1(lstm1_input,
                                              (h1_prev, c1_prev) if h1_prev is not None else None)
            h1_prev, c1_prev = h1_n, c1_n

            # Compute attention and context vector (w_t)
            alpha_beta_kappa = self.alpha_beta_kappa(h1_out.squeeze(1))
            alpha_hat, beta_hat, kappa_hat = torch.chunk(
                alpha_beta_kappa, 3, dim=1)  # [batch, window_K]
            alpha, beta, kappa = torch.exp(alpha_hat), torch.exp(
                beta_hat), kappa_prev + torch.exp(kappa_hat) if kappa_prev is not None else torch.exp(kappa_hat)
            # [batch, alphabet_size]
            # [batch, alphabet_size]
            w_t = self.compute_phi(alpha, beta, kappa, c_seq)

            # Forward pass through LSTM2
            lstm2_input = torch.cat(
                [x, w_t.unsqueeze(1), h1_out], dim=2)
            h2_out, (h2_n, c2_n) = self.lstm2(lstm2_input,
                                              (h2_prev, c2_prev) if h2_prev is not None else None)
            h2_prev, c2_prev = h2_n, c2_n

            # Output layer
            out = self.fc2(h2_out.squeeze(1))  # [batch, output_size]

            # Split and transform the output into GMM parameters
            pi, mu1, mu2, std1, std2, corr, eot = self.mdn(out.unsqueeze(1))

            # Sample from the GMM
            sample = sample_from_gmm(pi, mu1, mu2, std1, std2, corr)

            # Prepare the sample for the next input
            # Should return a tensor of shape [batch, 2]
            # print("sample", sample.shape)
            new_x = process_sample_for_next_input(sample)
            # new_x = sample

            # Update x for the next iteration, adding the eot probability
            # new_x is [batch, 2], eot is [batch, 1]
            eot_binary = (torch.sigmoid(eot) > threshold).float()
            x = torch.cat((eot_binary, new_x), dim=-
                          1)  # [batch, 3]
            # print("x", x.shape)

            # Append the new sample to predictions
            predictions.append(x.squeeze(1).detach())

        print("prediction", predictions[0].shape)
        predictions = torch.stack(predictions, axis=0)
        print("predictions", predictions.shape)
        predictions = predictions.permute(1, 0, 2)
        print("predictions", predictions.shape)
        # predictions[0][0][0] = 1.0

        return predictions


def sample_from_gmm(pi, mu1, mu2, std1, std2, corr):
    # Assume pi, mu1, mu2, std1, std2, corr are [batch, 1, K]
    # Flatten the pi tensor to [batch, K] for sampling
    # Find the index of the most likely component for each instance in the batch
    _, most_likely_idx = pi.squeeze(1).max(dim=1)

    # Adjust indexing for batched operation
    batch_indices = torch.arange(mu1.size(0), device=mu1.device)

    # Extract the means for the most likely Gaussian component
    mu = torch.stack([mu1[batch_indices, 0, most_likely_idx],
                      mu2[batch_indices, 0, most_likely_idx]], dim=-1)

    # take the mean of the most probable Gaussian component as the representative point.
    return mu.squeeze(1)


def process_sample_for_next_input(sample):
    return sample.unsqueeze(0)  # add a batch dimension


def sample_from_mdn(mdn_output):
    pi, mu1, mu2, sigma1, sigma2, rho, eos_prob = mdn_output
    batch_size, seq_len, num_mixtures = pi.size()

    mixture_idx = Categorical(pi).sample()  # [batch, seq_len]

    print("mu1", mu1.shape)
    print("mixture_idx", mixture_idx)
    mu1_selected = torch.gather(
        mu1, 2, mixture_idx.unsqueeze(-1)).squeeze(-1)  # [batch, seq_len]
    mu2_selected = torch.gather(mu2, 2, mixture_idx.unsqueeze(-1)).squeeze(-1)
    sigma1_selected = torch.gather(
        sigma1, 2, mixture_idx.unsqueeze(-1)).squeeze(-1)
    sigma2_selected = torch.gather(
        sigma2, 2, mixture_idx.unsqueeze(-1)).squeeze(-1)
    rho_selected = torch.gather(rho, 2, mixture_idx.unsqueeze(-1)).squeeze(-1)
    print("mu1_selected", mu1_selected.shape)

    cov = torch.zeros((batch_size, seq_len, 2, 2), device=mu1.device)
    cov[:, :, 0, 0] = sigma1_selected ** 2
    cov[:, :, 1, 1] = sigma2_selected ** 2
    cov[:, :, 0, 1] = cov[:, :, 1, 0] = rho_selected * \
        sigma1_selected * sigma2_selected
    print("cov", cov.shape)

    print("torch.stack((mu1_selected, mu2_selected), dim=-1)",
          torch.stack((mu1_selected, mu2_selected), dim=-1).shape)
    bivariate_normal = MultivariateNormal(torch.stack(
        (mu1_selected, mu2_selected), dim=-1), covariance_matrix=cov)
    next_x = bivariate_normal.sample()

    print("next_x", next_x.shape)
    print("eos_prob", eos_prob.shape)
    eos = torch.distributions.Bernoulli(eos_prob).sample()
    print("eos", eos.shape)

    next_point = torch.cat([eos, next_x], dim=2)  # [batch, seq_len, 3]

    return next_point
