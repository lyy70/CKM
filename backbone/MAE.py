import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.models_mae import MaskedAutoencoderViT  # 导入MAE模型
from utils.pos_embed import interpolate_pos_embed

class MAE_Encoder(nn.Module):
    def __init__(self, pretrained_path):
        super(MAE_Encoder, self).__init__()

    def forward(self, x, mask):
        B, N, C = x.shape
        x = self.vit.patch_embed.proj(x)
        x = x * (1 - mask.unsqueeze(-1))
        x = self.vit.blocks(x)
        return x

# 3. MAE 解码器（重构输入特征）
class MAE_Decoder(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=512, num_layers=4, num_heads=8):
        super(MAE_Decoder, self).__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 512)

    def forward(self, encoded_features, mask):
        decoded = self.decoder(encoded_features, encoded_features)
        decoded = self.fc(decoded)
        return decoded

class MAE(nn.Module):
    def __init__(self, mask_ratio, input_size, pretrained_path):
        super(MAE, self).__init__()
        self.mask_generator = RandomMaskingGenerator(input_size=input_size, mask_ratio=mask_ratio)
        self.encoder = MAE_Encoder(pretrained_path=pretrained_path)
        self.decoder = MAE_Decoder()

    def forward(self, x):
        mask = self.mask_generator().to(x.device)
        encoded_features = self.encoder(x, mask)
        reconstructed_features = self.decoder(encoded_features, mask)
        return reconstructed_features, mask
