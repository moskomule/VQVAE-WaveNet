import copy
from typing import Tuple, Callable, Optional, Any

import torch
from homura import is_distributed, get_world_size
from homura.modules import custom_straight_through_estimator
from torch import nn
from torch.distributed import all_reduce
from torch.nn import functional as F


class EMAModel(nn.Module):
    # module that tracks exponential moving average
    def __init__(self,
                 model: nn.Module,
                 gamma: float
                 ) -> None:
        super().__init__()
        self.model = model
        self.decay = gamma

        ema_model = copy.deepcopy(model)
        for param in ema_model.parameters():
            param.requires_grad_(False)
        self.ema_model = ema_model

    def forward(self, *args, **kwargs) -> Any:
        if not self.training:
            return self.ema_model(*args, **kwargs)

        # if training
        self.ema_update()
        return self.model(*args, **kwargs)

    def ema_update(self):
        for param, e_param in zip(self.model.parameters(), self.ema_model.parameters()):
            exponential_moving_average_(e_param.data, param.data, self.gamma)


class Encoder(nn.Sequential):
    # encoder
    def __init__(self,
                 dim: int
                 ) -> None:
        super().__init__(nn.Conv1d(1, dim, 4, 2, 1),
                         nn.ReLU(),

                         nn.Conv1d(dim, dim, 4, 2, 1),
                         nn.ReLU(),

                         nn.Conv1d(dim, dim, 4, 2, 1),
                         nn.ReLU(),

                         nn.Conv1d(dim, dim, 4, 2, 1),
                         nn.ReLU(),

                         nn.Conv1d(dim, dim, 4, 2, 1),
                         nn.ReLU(),

                         nn.Conv1d(dim, dim, 4, 2, 1),
                         )


class ConditionEmbed(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_global_cond: int,
                 global_emb_dim: int,
                 local_emb_dim: int,
                 upscale_factor: int = 64
                 ):
        super().__init__()
        self.local_emb = nn.Sequential(nn.Conv1d(input_dim, local_emb_dim, 3, padding=1, dilation=1),
                                       nn.ReLU(),

                                       nn.Conv1d(local_emb_dim, local_emb_dim, 3, padding=2, dilation=2),
                                       nn.ReLU(),

                                       nn.Conv1d(local_emb_dim, local_emb_dim, 3, padding=4, dilation=4),
                                       nn.ReLU(),

                                       nn.Conv1d(local_emb_dim, local_emb_dim, 3, padding=8, dilation=8),
                                       nn.ReLU(),

                                       nn.Conv1d(local_emb_dim, local_emb_dim, 3, padding=16, dilation=16),
                                       )
        self.global_emb = nn.Embedding(num_global_cond, global_emb_dim)
        self.upscale_factor = upscale_factor

    def forward(self,
                local_condition: torch.Tensor,
                global_condition: torch.Tensor
                ) -> torch.Tensor:
        local_condition = self.local_emb(local_condition)
        local_condition = F.interpolate(local_condition, self.upscale_factor * local_condition.size(-1),
                                        mode='bilinear')

        global_condition = self.global_emb(global_condition).unsqueeze(-1)
        global_condition = F.interpolate(global_condition, local_condition.size(-1))
        return torch.cat([local_condition, global_condition], dim=1)


def _torch_knn(keys: torch.Tensor,
               queries: torch.Tensor,
               num_neighbors: int,
               distance: str
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        if distance == "dot_product":
            scores = keys.mm(queries.t())
        else:
            scores = keys.mm(queries.t())
            scores *= 2
            scores -= (keys.pow(2)).sum(1, keepdim=True)
            scores -= (queries.pow(2)).sum(1).unsqueeze_(0)
        scores, indices = scores.topk(k=num_neighbors, dim=0, largest=True)
        scores = scores.t()
        indices = indices.t()

    return scores, indices


def k_nearest_neighbor(keys: torch.Tensor,
                       queries: torch.Tensor,
                       num_neighbors: int,
                       distance: str, *,
                       backend: Optional[str] = "torch"
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """ k-Nearest Neighbor search
    :param keys: tensor of (num_keys, dim)
    :param queries: tensor of (num_queries, dim)
    :param num_neighbors: `k`
    :param distance: name of distance (`dot_product` or `l2`)
    :param backend: backend (`faiss` or `torch`)
    :return: scores, indices
    """

    assert distance in {'dot_product', 'l2'}
    assert keys.size(1) == queries.size(1)
    f = _torch_knn
    return f(keys, queries, num_neighbors, distance)


def exponential_moving_average_(base: torch.Tensor,
                                update: torch.Tensor,
                                momentum: float
                                ) -> torch.Tensor:
    return base.mul_(momentum).add_(update, alpha=1 - momentum)


class VQModule(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 dict_size: int,
                 momentum: float,
                 eps: float = 1e-5,
                 ) -> None:
        super(VQModule, self).__init__()
        self.emb_dim = emb_dim
        self.dict_size = dict_size
        self.momentum = momentum
        self.eps = eps

        embed = torch.randn(self.dict_size, self.emb_dim)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(self.dict_size))
        self.register_buffer('embed_avg', self.embed.T.clone())

    def forward(self,
                input: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        quantized, ids = self._quantize(input)
        commit_loss = F.mse_loss(input, quantized)
        quantized = custom_straight_through_estimator(quantized, input)
        return quantized, commit_loss, ids

    @torch.no_grad()
    def _quantize(self,
                  input: torch.Tensor
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        is_2d = (input.ndim == 2)
        if is_2d:
            flatten = input
        else:
            # flatten (BHW)xC
            flatten = input.transpose(1, -1).reshape(-1, self.emb_dim)
        # dist: (BHW)x1, ids: (BHW)x1
        _, ids = k_nearest_neighbor(self.embed, flatten, 1, 'l2')
        # embed_onthot: (BHW)x{dict_size}
        embed_onehot = F.one_hot(ids.view(-1), self.dict_size).to(flatten.dtype)
        if not is_2d:
            # quantized: -> BxCxHxW, ids: -> BxHxW
            b, _, h, w = input.size()
            ids = ids.view(b, h, w)
        quantized = self.lookup(ids)

        if self.training:
            # embed_onehot_sum: {dict_size}
            embed_onehot_sum = embed_onehot.sum(dim=0)
            # embed_sum: Cx{dict_size}
            embed_sum = flatten.T @ embed_onehot
            if is_distributed():
                all_reduce(embed_onehot)
                all_reduce(embed_sum)
                ws = get_world_size()
                embed_onehot /= ws
                embed_sum /= ws

            exponential_moving_average_(self.cluster_size, embed_onehot_sum, self.momentum)
            exponential_moving_average_(self.embed_avg, embed_sum, self.momentum)
            n = self.cluster_size.sum()
            cluster_size = n * (self.cluster_size + self.eps) / (n + self.dict_size * self.eps)
            self.embed.copy_(self.embed_avg.T / cluster_size.unsqueeze(1))
        return quantized, ids

    def lookup(self,
               ids: torch.Tensor
               ) -> torch.Tensor:
        return F.embedding(ids, self.embed).transpose(1, -1)


class VQVAE(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 condition_embed: ConditionEmbed,
                 dict_size: int,
                 emb_dim: int,
                 beta: float,
                 momentum: float,
                 loss_func: Callable
                 ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vq = VQModule(emb_dim, dict_size, momentum)
        self.condition_embed = condition_embed
        self.loss_func = loss_func
        self.beta = beta

    def forward(self,
                x_enc: torch.Tensor,
                x_dec: torch.Tensor,
                global_condition: torch.Tensor,
                target: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x_enc)
        e, loss2, _ = self.vq(z)
        local_condition = e
        condition = self.condition_embed(local_condition, global_condition)
        y = self.decoder(x_dec, condition)

        loss1 = self.loss_func(y, target)
        return loss1, loss2
