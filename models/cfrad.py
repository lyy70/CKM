import gc
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from backbone.CNN import CNN
from models.utils.continual_model import ContinualModel
from utils.adjust_lr import adjust_learning_rate, WarmCosineScheduler
from utils.resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from utils.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from utils.loss import loss_fucntion
from torch.nn import CosineSimilarity
from typing import List
from utils.StableAdamW import StableAdamW

cos = CosineSimilarity(dim=1)

class CFRAD(ContinualModel):
    def __init__(self, args):
        super(CFRAD, self).__init__(args)
        # self.nets = []
        self.nets = nn.ModuleList()  # 改成 ModuleList
        self.encoder, _ = resnet18(pretrained=True)
        self.encoder.eval()
        # for p in self.encoder.parameters():
        #     p.requires_grad = False
        self.current_task = -1
        self.t_c_arr = []
        self.item_list = self.args.item_list
        self.param_weights_bn = []
        self.param_basis_bn = []
        self.param_weights_decoder = []
        self.param_basis_decoder = []
        self.threshold = self.args.threshold

    def begin_il(self, dataset):
        self.t_c_arr = dataset.t_c_arr

        for item in tqdm(self.item_list):
            _, bn = resnet18(pretrained=False)
            bn.train()
            decoder = de_resnet18(pretrained=False)
            decoder.train()
            net = CNN(bn, decoder)
            self.nets.append(net)
            print(f"\n===== Initialized network for item '{item}'initialized successfully.=====")

    def train_model(self, train_loader, item):
        self.current_task += 1
        categories = self.t_c_arr[self.current_task]
        self.encoder = self.encoder.to(self.device)

        for category in categories:
            # os.makedirs('./checkpoints/visa/', exist_ok=True)
            # weight_path = f'./checkpoints/visa/{item}_{category}_model.pth'
            # decomp_path = f'./checkpoints/visa/{item}_{category}_decomp.pth'

            # if os.path.exists(weight_path) and os.path.exists(decomp_path):
            #     print(f"⚠️ {item}, {category} 已存在，跳过该类别训练")
            #     continue
            # ====== 正常训练 ======
            network = self.nets[category].to(self.device)
            network.train()
            optimizer = StableAdamW(network.parameters(), lr=self.args.lr, betas=(0.9, 0.95))
            lr_scheduler = WarmCosineScheduler(optimizer, base_value=1e-3, final_value=1e-4,
                                               total_iters=self.args.epochs * len(train_loader),
                                               warmup_iters=100)

            for epoch in range(self.args.epochs):
                adjust_learning_rate(optimizer, epoch)
                loss_list = []
                for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
                    img = data[0].to(self.device)
                    inter_feature = self.encoder(img)
                    rec_featuer = network(inter_feature)
                    # loss = loss_fucntion(inter_feature, rec_featuer)
                    loss = loss_fucntion(inter_feature, rec_featuer, y=3)
                    loss.backward()
                    nn.utils.clip_grad_norm(network.parameters(), max_norm=0.1)
                    optimizer.step()
                    loss_list.append(loss.item())
                    lr_scheduler.step()
                print('epoch [{}/{}], loss:{:.3f}'.format(epoch + 1, self.args.epochs, np.mean(loss_list)))
            # torch.save(network.state_dict(), weight_path)
            # print(f"✅ 权重已保存: {weight_path}")

            inter_feature_list = [[] for _ in range(3)]
            bn_feature_list = []
            with torch.no_grad():
                for i, data in enumerate(train_loader):
                    img = data[0].to(self.device)
                    self.encoder = self.encoder.to(self.device)
                    inter_feature = self.encoder(img)
                    for j in range(3):
                        inter_feature_list[j].append(inter_feature[j])
                    bn_feature = network.bn(inter_feature)
                    bn_feature_list.append(bn_feature)

            # ==== SVD for BN ====
            for j in range(len(inter_feature_list)):
                inter_feature_list[j] = torch.cat(inter_feature_list[j], dim=0)
            weights_bn, basis_bn= self.decomposition_net_bn(network.bn, inter_feature_list)#sampled_inter_features, inter_feature_list

            # ==== SVD for Decoder ====
            bn_feature_all = torch.cat(bn_feature_list, dim=0)
            weights_decoder, basis_decoder = self.decomposition_net(network.decoder, bn_feature_all)

            self.param_weights_bn.append(weights_bn)
            self.param_basis_bn.append(basis_bn)
            self.param_weights_decoder.append(weights_decoder)
            self.param_basis_decoder.append(basis_decoder)

            # # 保存到本地文件
            # torch.save({
            #     "weights_bn": weights_bn,
            #     "basis_bn": basis_bn,
            #     "weights_decoder": weights_decoder,
            #     "basis_decoder": basis_decoder
            # }, decomp_path)
            # print(f"✅ 模型参数已保存: {decomp_path}")

        ##########################################################
        # def tensor_size_bytes(tensor):
        #     return tensor.numel() * tensor.element_size()
        #
        # def estimate_storage(param_list):
        #     total_bytes = 0
        #     for w in param_list:
        #         if isinstance(w, (list, tuple)):  # 处理 SVD 返回的 U, S, V
        #             for mat in w:
        #                 total_bytes += tensor_size_bytes(mat)
        #         else:
        #             total_bytes += tensor_size_bytes(w)
        #     return total_bytes
        #
        # total_bytes_before = 0
        # for network in self.nets:
        #     for param in network.parameters():  # 遍历所有参数
        #         total_bytes_before += tensor_size_bytes(param)
        # total_MB_before = total_bytes_before / (1024 ** 2)
        # print("Estimated storage for original model parameters: {:.2f} MB".format(total_MB_before))
        #
        # total_bytes_after = 0
        # total_bytes_after += estimate_storage(self.param_weights_bn)
        # total_bytes_after += estimate_storage(self.param_basis_bn)
        # total_bytes_after += estimate_storage(self.param_weights_decoder)
        # total_bytes_after += estimate_storage(self.param_basis_decoder)
        # total_MB_after = total_bytes_after / (1024 ** 2)
        # print("Estimated storage for CKM parameters after SVD: {:.2f} MB".format(total_MB_after))
        ##########################################################

    def decomposition_net_bn(self, bn_net, example_data_list):
        self.net = bn_net
        extracted_features = []
        def hook_fn(module, input, output):
            extracted_features.append(output.clone().detach()) #extracted_features.append(output), extracted_features.append(output.clone().detach())
        hook_handles = []
        hook_layers = [self.net.conv1, self.net.conv2, self.net.conv3]
        for layer in hook_layers:
            handle = layer.register_forward_hook(hook_fn)
            hook_handles.append(handle)
        ################## mvtec加速 ######
        # _ = self.net(example_data_list)
        #######################visa#################
        batch_size = 64
        for i in range(0, example_data_list[0].size(0), batch_size):
            input_batch = [feat[i:i + batch_size].to(self.device) for feat in example_data_list]
            _ = self.net(input_batch)
        ########################
        for handle in hook_handles:
            handle.remove()

        self.feature_list_bn = []
        mat_list = self.get_representation_matrix(extracted_features)
        self.feature_list_bn = self.update_Basis(mat_list, self.threshold, self.feature_list_bn)
        # self.param_basis_bn.append(self.feature_list_bn)
        weights = []
        for k, layer in enumerate(hook_layers):
            C = self.feature_list_bn[k].size(0)
            for name, params in layer.named_parameters():
                if 'weight' in name and params.requires_grad:
                    try:
                        W = params.data.view(C, -1)
                    except RuntimeError:
                        continue
                    param_weight = torch.mm(self.feature_list_bn[k].transpose(0, 1), W)
                    weights.append(param_weight)

        return weights, self.feature_list_bn

    def decomposition_net(self, decoder_net, example_data):
        self.net = decoder_net
        ####################mvtec##############
        # feature_list = self.net(example_data)
        # mat_list = self.get_representation_matrix(feature_list)
        ########################visa###############
        all_feature_list = None
        batch_size = 64
        # 分批送入
        with torch.no_grad():
            for i in range(0, example_data.size(0), batch_size):
                batch = example_data[i:i + batch_size].to(self.device)
                feature_list = self.net(batch)  # 每次是 List[Tensor]
                if all_feature_list is None:
                    all_feature_list = [[] for _ in range(len(feature_list))]
                for j in range(len(feature_list)):
                    all_feature_list[j].append(feature_list[j].cpu())  # 拿到 CPU
                del batch, feature_list
                torch.cuda.empty_cache()
                gc.collect()

        merged_feature_list = [torch.cat(layer_feats, dim=0) for layer_feats in all_feature_list]
        mat_list = self.get_representation_matrix(merged_feature_list)
        ##########################
        # update basis
        self.feature_list = []
        self.feature_list = self.update_Basis(mat_list, self.threshold, self.feature_list)
        # self.param_basis_decoder.append(self.feature_list)
        # cal weights of current task
        weights = []
        layers = [self.net.layer3, self.net.layer2, self.net.layer1]
        for k, layer in enumerate(layers):
            C = self.feature_list[k].size(0)  # 通道数
            for block in layer:
                for name, params in block.named_parameters():
                    if 'weight' in name and params.requires_grad:
                        try:
                            W = params.data.view(C, -1)
                        except RuntimeError:
                            continue
                        param_weight = torch.mm(self.feature_list[k].transpose(0, 1), W)
                        weights.append(param_weight)

        return weights, self.feature_list

    def get_representation_matrix(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        mat_final = []
        for feat in features:
            B, C, H, W = feat.shape
            mat = feat.permute(1, 0, 2, 3).contiguous().view(C, -1)
            mat_final.append(mat)
        return mat_final

    def update_Basis(self, mat_list, threshold, feature_list=[]):
        if not feature_list:
            for i in range(len(mat_list)):
                activation = mat_list[i].detach().cpu().numpy()
                U, S, Vh = np.linalg.svd(activation, full_matrices=False)

                sval_total = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total
                r = np.sum(np.cumsum(sval_ratio) < threshold)  #
                basis = torch.Tensor(U[:, 0:r]).to(self.device)
                feature_list.append(basis)

        else:
            for i in range(len(mat_list)):
                activation = mat_list[i].detach().cpu().numpy()
                basis = feature_list[i].detach().cpu().numpy()

                U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
                sval_total = (S1 ** 2).sum()

                act_hat = activation - np.dot(np.dot(
                    basis,
                    basis.transpose()
                ), activation)
                U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)

                sval_hat = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total
                accumulated_sval = (sval_total - sval_hat) / sval_total

                r = 0
                for ii in range(sval_ratio.shape[0]):
                    if accumulated_sval < threshold:
                        accumulated_sval += sval_ratio[ii]
                        r += 1
                    else:
                        break
                if r == 0:
                    print('Skip Updating Basis for layer: {}'.format(i + 1))
                    continue
                # update GPM
                Ui = np.hstack((basis, U[:, 0:r]))
                if Ui.shape[1] > Ui.shape[0]:
                    feature_list[i] = torch.Tensor(Ui[:, 0:Ui.shape[0]]).to(self.device)
                else:
                    feature_list[i] = torch.Tensor(Ui).to(self.device)
        self.basis_size = 0
        for i in range(len(feature_list)):
            self.basis_size += feature_list[i].shape[1] * feature_list[i].shape[0]
        return feature_list

    # def load_all_decomps(self):
    #     # 清空已有列表
    #     self.param_weights_bn.clear()
    #     self.param_basis_bn.clear()
    #     self.param_weights_decoder.clear()
    #     self.param_basis_decoder.clear()
    #
    #     decomp_files = glob.glob('./checkpoints/*_decomp.pth')
    #
    #     def extract_category(path):
    #         match = re.search(r'_(\d+)_decomp\.pth$', path)
    #         return int(match.group(1)) if match else -1
    #
    #     decomp_files.sort(key=extract_category)
    #
    #     for file_path in decomp_files:
    #         category = extract_category(file_path)
    #         decomp_data = torch.load(file_path, map_location=self.device)
    #         self.param_weights_bn.append(decomp_data["weights_bn"])
    #         self.param_basis_bn.append(decomp_data["basis_bn"])
    #         self.param_weights_decoder.append(decomp_data["weights_decoder"])
    #         self.param_basis_decoder.append(decomp_data["basis_decoder"])
    #         print(f"✅ 已加载 {file_path} (class {category})")
    #     print(f"✅ 总共加载 {len(decomp_files)} 个任务的分解参数")

    # def forward(self, x: torch.Tensor, class_idx: int):
    #     # 确认 decomps 已加载
    #     if not hasattr(self, 'decomps_loaded') or not self.decomps_loaded:
    #         self.load_all_decomps()
    #         self.decomps_loaded = True
    #
    #     inputs = x.to(self.device)
    #     inter_feature = self.encoder(inputs)
    #     inter_feature_detach = [f.to(self.device).detach() for f in inter_feature]
    #
    #     with torch.no_grad():
    #         weights = self.param_weights_decoder[class_idx]
    #         basis_list = self.param_basis_decoder[class_idx]
    #         weights_bn = self.param_weights_bn[class_idx]
    #         basis_list_bn = self.param_basis_bn[class_idx]
    #
    #         self.composition_net_bn(self.nets[class_idx].bn, weights_bn, basis_list_bn)
    #         self.composition_net(self.nets[class_idx].decoder, weights, basis_list)
    #
    #         network = self.nets[class_idx].to(self.device)
    #         network.eval()
    #         rec_feature = network(inter_feature)
    #         network.cpu()
    #         rec_feature_detach = [f.detach() for f in rec_feature]
    #
    #     return inter_feature_detach, rec_feature_detach

    def forward(self, x: torch.Tensor):
        inputs = x.to(self.device)
        inter_feature = self.encoder(inputs)

        best_rec_feature = None
        best_score = float('inf')

        for class_idx in range(self.current_task + 1):
            weights = self.param_weights_decoder[class_idx]
            basis_list = self.param_basis_decoder[class_idx]
            weights_bn = self.param_weights_bn[class_idx]
            basis_list_bn = self.param_basis_bn[class_idx]

            self.composition_net_bn(self.nets[class_idx].bn, weights_bn, basis_list_bn)
            self.composition_net(self.nets[class_idx].decoder, weights, basis_list)

            network = self.nets[class_idx].to(self.device)
            network.eval()
            rec_feature = network(inter_feature)
            network.cpu()
            score = 0
            for i in range(len(inter_feature)):
                fs = inter_feature[i]
                ft = rec_feature[i]
                score += torch.mean(1 - F.cosine_similarity(fs, ft, dim=1))
            if score < best_score:
                best_score = score
                best_rec_feature = rec_feature
        return inter_feature, best_rec_feature

    ###############################如果知道索引，可加速测试#####################
    # def forward(self, x: torch.Tensor, class_idx: int) -> torch.Tensor:
    #     inputs = x.to(self.device)
    #     inter_feature = self.encoder(inputs)
    #
    #     with torch.no_grad():
    #         weights = self.param_weights_decoder[class_idx]
    #         basis_list = self.param_basis_decoder[class_idx]
    #         weights_bn = self.param_weights_bn[class_idx]
    #         basis_list_bn = self.param_basis_bn[class_idx]
    #
    #         self.composition_net_bn(self.nets[class_idx].bn, weights_bn, basis_list_bn)
    #         self.composition_net(self.nets[class_idx].decoder, weights, basis_list)
    #
    #         network = self.nets[class_idx].to(self.device)
    #         network.eval()
    #         rec_feature = network(inter_feature)
    #         network.cpu()
    #     return inter_feature, rec_feature

    def composition_net(self, decoder_net, weights, basis_list):
        self.net = decoder_net
        layers = [self.net.layer3, self.net.layer2, self.net.layer1]
        weight_idx = 0
        for k, layer in enumerate(layers):
            basis = basis_list[k]  # 基底 (C, K)
            # print(f"Layer {k + 1} basis shape: {basis.shape}")  # 打印basis形状
            for block in layer:
                for name, params in block.named_parameters():
                    if 'weight' in name and params.requires_grad:
                        param_weight = weights[weight_idx]  # shape (K, N)
                        # print(f"Weight idx {weight_idx} param_weight shape: {param_weight.shape}")
                        weight_idx += 1
                        W_recon = torch.mm(basis, param_weight)
                        orig_shape = params.data.shape  # e.g. (out_channels, in_channels, kH, kW)
                        W_recon = W_recon.view(orig_shape).to(params.device)
                        params.data.copy_(W_recon)

    def composition_net_bn(self, bn_net, weights_bn, basis_list_bn):
        self.net = bn_net
        layers = [self.net.conv1, self.net.conv2, self.net.conv3]
        weight_idx = 0
        for k, layer in enumerate(layers):
            basis = basis_list_bn[k]  # 当前层的基底 (C, K)
            for name, params in layer.named_parameters():
                if 'weight' in name and params.requires_grad:
                    param_weight = weights_bn[weight_idx]  # (K, N)
                    weight_idx += 1
                    W_recon = torch.mm(basis, param_weight)  # (C, N)
                    W_recon = W_recon.view_as(params.data).to(params.device)
                    params.data.copy_(W_recon)













