import time
import cv2
from utils.conf import base_visa_path
from utils.conf import base_mvtec_path
from utils.loggers import *
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from models import get_model
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets import get_dataset
import numpy as np


def apply_ad_scoremap(image, scoremap, alpha=0.6):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)
def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)

def evaluate(model: ContinualModel, dataset: ContinualDataset, item, item_lists, save_path, last=False) -> Tuple[float, float]:
    save_images = (item == item_lists[-1])
    # # --- 持久化历史指标：只初始化一次 ---
    # if not hasattr(model, "fm_img_auc"):
    #     model.fm_img_auc = {name: [] for name in item_lists}  # 图像级 AUROC 历史
    # if not hasattr(model, "fm_pix_ap"):
    #     model.fm_pix_ap = {name: [] for name in item_lists}   # 像素级 AP 历史
    # # 补齐当前 item_lists 中所有类别的键
    # for name in item_lists:
    #     model.fm_img_auc.setdefault(name, [])
    #     model.fm_pix_ap.setdefault(name, [])

    auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
    auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

    for k, test_loader in enumerate(dataset.test_loaders):#k(0,8).
        if last and k < len(dataset.test_loaders) - 1:
            continue

        key = item_lists[k] if k < len(item_lists) else f"task_{k}"

        model.fm_img_auc.setdefault(key, [])
        model.fm_pix_ap.setdefault(key, [])

        gt_list_px = []
        pr_list_px = []
        gt_list_sp = []
        pr_list_sp = []

        gaussian_kernel = get_gaussian_kernel(kernel_size=3, sigma=4).to(model.device)

        # preheat_batch = next(iter(test_loader))[0].to(model.device)
        # for _ in range(5):
        #     _ = model(preheat_batch)
        # batch_times = []

        for data in test_loader:
            imgs, cls_idx, gt, label, img_path = data[0], data[1], data[2], data[3], data[4]
            imgs = imgs.to(model.device)
            inputs, outputs = model(imgs)
            # start = time.time()
            # inputs, outputs = model(imgs)
            #############################
            # torch.cuda.synchronize()  # 等待GPU完成
            # end = time.time()
            # batch_time = end - start
            # batch_times.append(batch_time)
            ###############################
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, imgs.shape[-1])
            anomaly_map = gaussian_kernel(anomaly_map)#[1,1,256,256]
            ################# visualization ##################
            # if save_images:
            #     path = os.path.join(base_visa_path(), img_path[0])# base_mvtec_path, base_visa_path
            #     cls = path.split('/')[-2]
            #     filename = path.split('/')[-1]
            #     vis = cv2.cvtColor(cv2.resize(cv2.imread(path), (256, 256)), cv2.COLOR_BGR2RGB)
            #     mask = normalize(anomaly_map.squeeze().cpu().detach().numpy())
            #     vis = apply_ad_scoremap(vis, mask)
            #     vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            #     save_vis = os.path.join(save_path, 'imgs', item_lists[k], cls)
            #     os.makedirs(save_vis, exist_ok=True)
            #     cv2.imwrite(os.path.join(save_vis, filename), vis)
            ##############################################################################
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]
            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)
            gt_list_sp.append(label)
            anomaly_map = anomaly_map.flatten(1)
            sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * 0.01)]
            sp_score = sp_score.mean(dim=1)
            pr_list_sp.append(sp_score)

    #     ############### 计算平均FPS##################
    #     avg_batch_time = sum(batch_times) / len(batch_times)
    #     batch_size = imgs.size(0)
    #     fps = batch_size / avg_batch_time
    #     print(f"⚡ 当前测试任务 '{key}' 平均每批推理时间: {avg_batch_time:.4f}s, FPS: {fps:.2f}")
    #     #######################################
        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().detach().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().detach().numpy()

        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = ader_evaluator(pr_list_px, pr_list_sp, gt_list_px,
                                                                                  gt_list_sp)
    #     # --- 存储历史 ---
    #     model.fm_img_auc[item_lists[k]].append(float(auroc_sp))
    #     model.fm_pix_ap[item_lists[k]].append(float(ap_px))
    #
        auroc_sp_list.append(auroc_sp)
        ap_sp_list.append(ap_sp)
        f1_sp_list.append(f1_sp)
        auroc_px_list.append(auroc_px)
        ap_px_list.append(ap_px)
        f1_px_list.append(f1_px)
        aupro_px_list.append(aupro_px)

        print(f"Train Class: {item} | Test Class: {item_lists[k]}")

        print(
            '{}: I-Auroc:{:.3f}, I-AP:{:.3f}, I-F1:{:.3f}, P-AUROC:{:.3f}, P-AP:{:.3f}, P-F1:{:.3f}, P-AUPRO:{:.3f}'.format(
                item_lists[k], auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))
    print(
        'Mean: I-Auroc:{:.3f}, I-AP:{:.3f}, I-F1:{:.3f}, P-AUROC:{:.3f}, P-AP:{:.3f}, P-F1:{:.3f}, P-AUPRO:{:.3f}'.format(
            np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
            np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))

    # # --- 计算 FM（安全判断：至少两次观测才计算） ---
    # num_tasks = len(dataset.test_loaders)  # 期望的列数（每个 task 对应一列）
    # # 用 np.nan 占位，保证行长度一致
    # row_img = [np.nan] * num_tasks   # image-level AUROC (对应 auroc_sp)
    # row_pix = [np.nan] * num_tasks   # pixel-level AP   (对应 ap_px)
    # # --- 保存 acc_matrix (任务间性能矩阵) ---
    # if not hasattr(model, "acc_matrix_img"):
    #     model.acc_matrix_img = []  # 保存 image-level AUROC
    # if not hasattr(model, "acc_matrix_pix"):
    #     model.acc_matrix_pix = []  # 保存 pixel-level AP
    #
    # # 每次 evaluate 结束，把一行性能结果加入矩阵
    # row_img = [scores for scores in auroc_sp_list]  # 当前模型在每个任务上的 AUROC
    # row_pix = [scores for scores in ap_px_list]     # 当前模型在每个任务上的 AP
    # model.acc_matrix_img.append(row_img)
    # model.acc_matrix_pix.append(row_pix)
    # # 如果之前的历史行长度可能不同，先把所有历史行 pad 到相同的最大列数
    # def _pad_rows(rows, target_len):
    #     for i, r in enumerate(rows):
    #         if len(r) < target_len:
    #             rows[i] = r + [np.nan] * (target_len - len(r))
    #     return rows
    # # 计算历史最大列数（考虑已有历史与当前行）
    # existing_img_rows = list(model.acc_matrix_img)
    # existing_pix_rows = list(model.acc_matrix_pix)
    # max_len = max([len(r) for r in (existing_img_rows + existing_pix_rows)] + [num_tasks])
    #
    # # pad 之前的历史
    # model.acc_matrix_img = _pad_rows(existing_img_rows, max_len)
    # model.acc_matrix_pix = _pad_rows(existing_pix_rows, max_len)
    #
    # # pad 当前行到 max_len（通常 current 行长度 == num_tasks，但以防万一）
    # if len(row_img) < max_len:
    #     row_img = row_img + [np.nan] * (max_len - len(row_img))
    # if len(row_pix) < max_len:
    #     row_pix = row_pix + [np.nan] * (max_len - len(row_pix))
    #
    # # append
    # model.acc_matrix_img.append(row_img)
    # model.acc_matrix_pix.append(row_pix)
    #
    # def compute_fm(acc_matrix):
    #     if not acc_matrix:
    #         return 0.0
    #     max_cols = max(len(r) for r in acc_matrix)
    #     R = len(acc_matrix)
    #     C = max_cols
    #
    #     mat = np.full((R, C), np.nan, dtype=float)
    #     for i, r in enumerate(acc_matrix):
    #         for j, v in enumerate(r):
    #             try:
    #                 mat[i, j] = float(v)
    #             except:
    #                 mat[i, j] = np.nan
    #     if R <= 1:
    #         return 0.0
    #     fm_values = []
    #     for j in range(C):
    #         hist = mat[:R - 1, j]  # 之前历史值
    #         last = mat[-1, j]  # 当前最后一行
    #         if np.isnan(last):
    #             continue
    #         if np.all(np.isnan(hist)):
    #             continue
    #         best = np.nanmax(hist)
    #         fm_values.append(best - last)
    #
    #     return float(np.nanmean(fm_values)) if fm_values else 0.0
    #
    # FM_img = compute_fm(model.acc_matrix_img)
    # FM_pix = compute_fm(model.acc_matrix_pix)
    #
    # print(f"Forgetting Measure (Image-level AUROC): {FM_img:.4f}")
    # print(f"Forgetting Measure (Pixel-level AP): {FM_pix:.4f}")

def train_il(args: Namespace) -> None:

    dataset = get_dataset(args)

    model = get_model(args)

    model.begin_il(dataset)

    item_lists = []

    for current_i, item in enumerate(args.item_list):

        train_loader = dataset.get_data_loaders()

        model.train_model(train_loader, item)

        item_lists.append(item)

        evaluate(model, dataset, item, item_lists, args.save_path)






