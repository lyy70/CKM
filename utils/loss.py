import torch
import torch.nn.functional as F
from functools import partial

# def loss_fucntion(a, b):
#     cos_loss = torch.nn.CosineSimilarity()
#     loss = 0
#     for item in range(len(a)):
#         loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
#                                       b[item].view(b[item].shape[0],-1)))
#     return loss

# def basis_regularization(basis_list):
#     loss = 0
#     for layer in basis_list:
#         A_delta = layer["A_delta"]
#         if A_delta.numel() > 0:
#             loss += torch.norm(A_delta, p=2)
#     return loss

def modify_grad_v2(x, factor):
    factor = factor.expand_as(x)  # [B, L, 1] -> [B, L, C]
    x = x * factor
    return x

def loss_fucntion(a, b, y):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1).detach()
        mean_dist = point_dist.mean()
        factor = (point_dist/mean_dist)**(y)
        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))
        partial_func = partial(modify_grad_v2, factor=factor)
        b_.register_hook(partial_func)

    loss = loss / len(a)
    return loss

# def loss_fucntion(a, b):
#     alpha = 0.8
#     cos = torch.nn.CosineSimilarity()
#     loss = 0
#     for i in range(len(a)):
#         cos_loss = torch.mean(1 - cos(a[i].flatten(1), b[i].flatten(1)))
#         l1_loss = torch.mean(torch.abs(a[i] - b[i]))
#         loss += alpha * cos_loss + (1 - alpha) * l1_loss
#     return loss
#
# def loss_fucntion(a, b):
#     loss = 0
#     for i in range(len(a)):
#         global_loss = torch.mean(1 - F.cosine_similarity(
#             a[i].view(a[i].shape[0], -1),
#             b[i].view(b[i].shape[0], -1)
#         ))
#
#         l2_loss = normalized_l2(a[i], b[i])
#         loss += 0.8 * global_loss + 0.2 * l2_loss
#     return loss
#
# def normalized_l2(a, b):
#     a_norm = F.normalize(a, dim=1)
#     b_norm = F.normalize(b, dim=1)
#     return torch.mean((a_norm - b_norm)**2)
