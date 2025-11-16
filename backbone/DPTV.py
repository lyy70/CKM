
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from utils.pos_embed import get_2d_sincos_pos_embed

class DPTV(nn.Module):
    def __init__(self, img_size=32, patch_size=2, in_chans=272,
                 embed_dim=240, divide_num=4, depth=12, num_heads=12,
                 decoder_embed_dim=240, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.divide_num = divide_num
        self.depth = depth

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.auxiliary_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):

        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.in_chans))
        return x

    def unpatchify(self, x):
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs

    def forward_encoder(self, x):
        x = self.patch_embed(x)#16,256,240
        x = x + self.pos_embed[:, 1:, :]#加上位置，去除额外添加一个 [CLS] token，
        ########################
        N, L, D = x.shape  # batch, length, dim
        assert L % self.divide_num == 0
        mask_ratio = 1 / self.divide_num# 每次仅保留 25% 的 token
        len_keep = int(L * mask_ratio)#随机选择64个tokens进行保留，其他192个tokens设置为0
        noise = torch.rand(N, L, device=x.device) #【16,256】
        ids_shuffle = torch.argsort(noise, dim=1)#对所有 token 进行升序排序
        ids_restore = torch.argsort(ids_shuffle, dim=1)#生成逆排序索引，ids_restore 用于后续恢复 token 顺序
        output = torch.zeros_like(x)
        ########################
        for i in range(self.divide_num):
            ids_keep = torch.cat([ids_shuffle[:, 0: i * len_keep], ids_shuffle[:, (i + 1) * len_keep:]], dim=-1) #所有被保留的 token的索引
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
            mask = torch.ones([N, L], device=x.device)
            mask[:, i * len_keep:(i + 1) * len_keep] = 0
            mask_bool_retain = mask > 0
            mask_bool_pad = mask < 0.5
            auxiliary_token = self.auxiliary_token.repeat(x_masked.shape[0], ids_restore.shape[1] - x_masked.shape[1], 1)#生成 辅助 token（auxiliary_token），用于填充被 mask 的部分，
            x_ = torch.cat([x_masked, auxiliary_token], dim=1)
            x_ = x_.masked_scatter_(mask_bool_retain.unsqueeze(-1).repeat(1, 1, x_.shape[2]), x_masked)
            x_ = x_.masked_scatter_(mask_bool_pad.unsqueeze(-1).repeat(1, 1, x_.shape[2]), auxiliary_token)
            x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
            #########################
            for blk in self.blocks:
                x_masked = blk(x_masked)

            x_masked = self.norm(x_masked)

            mask = torch.gather(mask, dim=1, index=ids_restore)
            mask_pad = mask > 0.5
            mask_pad = mask_pad.unsqueeze(-1).repeat(1, 1, x_masked.shape[2])
            x_masked = x_masked.masked_fill(mask_pad, 0)
            output = output + x_masked

        return output

    def forward_decoder(self, x):
        # embed tokens
        x = self.decoder_embed(x)
        # add pos embed
        x = x + self.decoder_pos_embed[:, 1:, :]
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        # predictor projection
        x = self.decoder_pred(x)
        # remove cls token
        return x

    def attn(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        ########################
        N, L, D = x.shape  # batch, length, dim
        assert L % self.divide_num == 0
        mask_ratio = 1 / self.divide_num
        len_keep = int(L * mask_ratio)
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        output = torch.zeros_like(x)
        ########################
        all_attn = 0
        for i in range(self.divide_num):
            ids_keep = torch.cat([ids_shuffle[:, 0: i * len_keep], ids_shuffle[:, (i + 1) * len_keep:]], dim=-1)
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
            mask = torch.ones([N, L], device=x.device)
            mask[:, i * len_keep:(i + 1) * len_keep] = 0
            mask_bool_retain = mask > 0
            mask_bool_pad = mask < 0.5
            auxiliary_token = self.auxiliary_token.repeat(x_masked.shape[0], ids_restore.shape[1] - x_masked.shape[1],1)
            x_ = torch.cat([x_masked, auxiliary_token], dim=1)
            x_ = x_.masked_scatter_(mask_bool_retain.unsqueeze(-1).repeat(1, 1, x_.shape[2]), x_masked)
            x_ = x_.masked_scatter_(mask_bool_pad.unsqueeze(-1).repeat(1, 1, x_.shape[2]), auxiliary_token)
            x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

            ###########
            for j, blk in enumerate(self.blocks):

                attn = blk.attention(x_masked)
                # attn[:, :, ids_keep[0], :] =0
                all_attn += attn
                x_masked = blk(x_masked)

            output = output + x_masked

        return all_attn /4

    def forward_loss(self, imgs, pred):

        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        dis_loss = (pred - target) ** 2
        dis_loss = dis_loss.mean(dim=-1)  # [N, L], mean loss per patch
        dir_loss = 1 - torch.nn.CosineSimilarity(-1)(pred, target)

        loss = 5 * dir_loss.mean() + dis_loss.mean()
        return loss

    def forward(self, imgs):
        #x[16,272,32,32]
        latent = self.forward_encoder(imgs)#[16,256,240]
        pred = self.forward_decoder(latent)#[16,256,1088]
        loss = self.forward_loss(imgs, pred)
        recon_feature = self.unpatchify(pred)#[16,272,32,32]
        return recon_feature, loss




