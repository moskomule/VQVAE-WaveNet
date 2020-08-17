""" PyTorch transplantation of https://github.com/dhgrs/chainer-VQ-VAE/blob/audio/WaveNet/modules.py
"""

from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class ResidualBlock(nn.Module):

    def __init__(self,
                 filter_size: int,
                 dilation: int,
                 residual_channels: int,
                 dilated_channels: int,
                 skip_channels: int,
                 condition_dim: int,
                 dropout_rate: float
                 ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(residual_channels, dilated_channels, filter_size,
                              padding=dilation * (filter_size - 1), dilation=dilation)
        self.condition_proj = nn.Conv1d(condition_dim, dilated_channels, 1)
        self.res = nn.Conv1d(dilated_channels // 2, residual_channels, 1)
        self.skip = nn.Conv1d(dilated_channels // 2, skip_channels, 1)

        self.filter_size = filter_size
        self.dilation = dilation
        self.residual_channels = residual_channels
        self.condition_dim = condition_dim
        self.dropout_rate = dropout_rate

        # will be initialized in self.initialize(n)
        self.register_buffer("queue", torch.empty(1))
        self.register_buffer("condition_queue", torch.empty(1))

    def forward(self,
                input: Tensor,
                condition: Tensor
                ) -> Tuple[Tensor, Tensor]:
        length = input.size(-1)

        if self.dropout_rate > 0:
            h = F.dropout(input, p=self.dropout_rate, training=self.training)
        else:
            h = input

        h = self.conv(h)
        h = h[:, :, :length]
        h += self.condition_proj(condition)

        # gated activation units
        tanh_z, sig_z = input.split(2, dim=1)
        z = tanh_z.tanh() * sig_z.sigmoid()

        # projection
        if input.size(-1) == z.size(-1):
            residual = self.res(z) + input
        else:
            residual = self.res(z) + input[:, :, -1:]

        skip = self.skip(z)
        return residual, skip

    def initialize(self,
                   n: int) -> None:
        self.conv.padding = 0
        self.queue = nn.Parameter(self.queue.new_zeros(n, self.residual_channels,
                                                       self.dilation * (self.filter_size - 1) + 1))
        self.condition_queue = nn.Parameter(self.condition_queue.new_zeros(n, self.condition_dim, 1))

    def pop(self) -> Tuple[Tensor, Tensor]:
        return self(self.queue, self.condition_queue)

    def push(self,
             input: Tensor,
             condition: Tensor
             ) -> None:
        self.queue = torch.cat([self.queue[:, :, 1:], input], dim=-1)
        self.condition_queue = torch.cat([self.condition_queue[:, :, 1:], condition], dim=-1)


class ResidualNet(nn.ModuleList):
    def __init__(self,
                 num_loops: int,
                 num_layers: int,
                 filter_size: int,
                 residual_channels: int,
                 dilated_channels: int,
                 skip_channels: int,
                 condition_dim: int,
                 dropout_rate: float
                 ) -> None:
        super().__init__()
        dilations = [2 ** i for i in range(num_layers)] * num_loops
        for dilation in dilations:
            self.append(ResidualBlock(filter_size, dilation, residual_channels,
                                      dilated_channels, skip_channels, condition_dim,
                                      dropout_rate))

    def forward(self,
                input: Tensor,
                condition: Tensor
                ) -> Tensor:
        for i, block in enumerate(self):
            if self.training:
                input, skip = block(input, condition)
            else:
                block.push(input, condition)
                input, skip = block.pop()
            if i == 0:
                skip_connection = skip
            else:
                skip_connection += skip
        return skip_connection

    def initialize(self,
                   n: int
                   ) -> None:
        for layer in self:
            layer.initialize(n)


class WaveNet(nn.Module):
    def __init__(self,
                 num_loops: int,
                 num_layers: int,
                 filter_size: int,
                 input_dim: int,
                 residual_channels: int,
                 dilated_channels: int,
                 skip_channels: int,

                 quantize: int,
                 use_logistic: bool,
                 num_mixture: bool,
                 log_scale_min: float,

                 condition_dim: int,

                 dropout_rate
                 ) -> None:
        super().__init__()
        output_dim = num_mixture if use_logistic else quantize
        self.embed = nn.Conv1d(input_dim, residual_channels, kernel_size=2, padding=1)
        self.resnet = ResidualNet(num_loops, num_layers, filter_size, residual_channels, dilated_channels,
                                  skip_channels, condition_dim, dropout_rate)
        self.proj1 = nn.Conv1d(skip_channels, skip_channels, 1)
        self.proj2 = nn.Conv1d(skip_channels, output_dim, 1)

        self.input_dim = input_dim
        self.quantize = quantize
        self.skip_channels = skip_channels
        self.register_buffer("log_scale_min", torch.empty(1).fill_(log_scale_min))
        self.register_buffer("epsilon", torch.empty(1).fill_(1e-12))
        self.register_buffer("embed_queue", torch.empty(1))
        self.register_buffer("proj1_queue", torch.empty(1))
        self.register_buffer("proj2_queue", torch.empty(1))

    def forward(self,
                input: Tensor,
                condition: Tensor
                ) -> Tensor:
        if self.training:
            # causal conv
            length = input.size(-1)
            x = self.embed(input)
            x = x[:, :, :length]

            # residual & skip-connection
            z = F.relu(self.resnet(x, condition))

            z = F.relu(self.proj1(z))
            return self.proj2(z)
        else:
            self.embed_queue = torch.cat([self.embed_queue[:, :, 1:], input], dim=-1)
            x = self.embed(self.embed_queue)
            x = F.relu(self.resnet(x, condition))

            self.proj1_queue = torch.cat([self.proj1_queue[:, :, 1:], x], dim=-1)
            x = F.relu(self.proj1(self.proj1_queue))

            self.proj2_queue = torch.cat([self.proj2_queue[:, :, 1:], x], dim=-1)
            x = self.proj2(self.proj2_queue)
            return x

    def logistic_loss(self,
                      y: Tensor,
                      t: Tensor
                      ) -> Tensor:
        nr_mix = y.size(1) // 3
        logit_probs = y[:, :nr_mix]
        means = y[:, nr_mix:2 * nr_mix]
        log_scales = y[:, 2 * nr_mix:3 * nr_mix]
        log_scales = torch.max(log_scales, self.log_scale_min)
        t = (127.5 * t).expand_as(means)
        centered_t = t - means
        inv_std = (-log_scales).exp()
        max_in = inv_std * (centered_t + 127.5 / (self.quantize - 1))
        cdf_max = max_in.sigmoid()
        min_in = inv_std * (centered_t - 127.5 / (self.quantize - 1))
        cdf_min = min_in.sigmoid()

        log_cdf_max = max_in - F.softplus(max_in)
        log_one_minus_cdf_min = -F.softplus(min_in)
        cdf_delta = cdf_max - cdf_min

        log_probs = torch.where(
            t < 127.5 * -0.999,
            log_cdf_max,
            torch.where(
                t > 127.5 * -0.999,
                log_one_minus_cdf_min,
                torch.max(cdf_delta, self.epsilon)
            )
        )

        log_probs = log_probs + logit_probs.log_softmax(dim=-1)
        return -log_probs.logsubexp(dim=1).mean()

    def initialize(self,
                   n: int) -> None:
        self.resnet.initialize(n)
        self.embed.pad = 0
        self.embed_queue = nn.Parameter(self.embed.new_zeros(n, self.input_dim, 2))
        self.proj1_queue = nn.Parameter(self.embed.new_zeros(n, self.skip_channels, 1))
        self.proj2_queue = nn.Parameter(self.embed.new_zeros(n, self.skip_channels, 1))
