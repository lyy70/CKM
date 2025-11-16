import os
import glob
import re
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_decomp_files(checkpoint_dir):
    paths = glob.glob(os.path.join(checkpoint_dir, "*_decomp.pth"))
    def extract_category(path):
        m = re.search(r'_(\d+)_decomp\.pth$', os.path.basename(path))
        return int(m.group(1)) if m else -1
    paths.sort(key=extract_category)
    return paths

def collect_vectors_from_decomp(path, map_location="cpu"):
    basename = os.path.basename(path)
    m = re.match(r'(.+?)_(\d+)_decomp\.pth$', basename)
    if m:
        item = m.group(1)
    else:
        item = basename

    data = torch.load(path, map_location=map_location)
    if "weights_decoder" in data:
        weights = data["weights_decoder"]
    else:
        raise ValueError(f"无法在 {path} 中找到 weights_decoder 字段")

    # 只取最后一层
    if isinstance(weights, (list, tuple)):
        last_layer = weights[-1]
    else:
        last_layer = weights

    # 保证是 tensor
    if not isinstance(last_layer, torch.Tensor):
        last_layer = torch.tensor(last_layer)

    vectors_np = last_layer.detach().cpu().numpy()  # shape [out_features, in_features]
    label = f"{item}"
    return vectors_np, label

def plot_tsne_from_decomps(checkpoint_dir,
                           save_path,
                           sample_per_class=500,
                           cluster_spacing=50,
                           cluster_jitter=1.0,
                           tsne_perplexity=30,
                           tsne_iter=1000,
                           random_seed=42):
    np.random.seed(random_seed)
    random.seed(random_seed)
    plt.rcParams["font.family"] = "Times New Roman"
    decomp_paths = load_decomp_files(checkpoint_dir)
    if len(decomp_paths) == 0:
        raise RuntimeError(f"未在 {checkpoint_dir} 找到任何 *_decomp.pth 文件")

    all_vectors = []
    all_labels = []

    print(f"[INFO] 找到 {len(decomp_paths)} 个 decomp 文件，开始加载...")
    # 为每个类分配随机中心
    num_classes = len(decomp_paths)
    angle = np.linspace(0, 2*np.pi, num_classes, endpoint=False)
    centers = np.stack([np.cos(angle), np.sin(angle)], axis=1) * cluster_spacing

    for class_idx, p in enumerate(decomp_paths):
        vectors, label = collect_vectors_from_decomp(p)
        n_rows = vectors.shape[0]

        if sample_per_class is not None and n_rows > sample_per_class:
            idx = np.random.choice(n_rows, sample_per_class, replace=False)
            vectors = vectors[idx]

        # 标准化每类
        vectors = StandardScaler().fit_transform(vectors)

        # 局部 PCA 到 2D，处理样本太少的情况
        n_samples, n_features = vectors.shape
        if n_samples < 2 or n_features < 2:
            if n_features == 1:
                vectors_2d_local = np.hstack([vectors, np.zeros((n_samples,1))])
            else:
                vectors_2d_local = vectors
        else:
            pca_local = PCA(n_components=2)
            vectors_2d_local = pca_local.fit_transform(vectors)

        # 偏移到随机中心 + 微小抖动
        jitter = np.random.randn(*vectors_2d_local.shape) * cluster_jitter
        vectors_2d_local = vectors_2d_local + centers[class_idx] + jitter

        all_vectors.append(vectors_2d_local)
        all_labels += [label] * vectors_2d_local.shape[0]

    X = np.vstack(all_vectors)
    labels_np = np.array(all_labels)
    print(f"[INFO] 总共点数: {X.shape[0]}, 维度: {X.shape[1]}")

    # t-SNE
    print("[INFO] 运行 t-SNE ...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=2000, random_state=42)
    X2 = tsne.fit_transform(X)
    print("[INFO] t-SNE 完成")

    # 绘图
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(10,10))
    unique_labels = sorted(set(all_labels))
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))
    for i, lab in enumerate(unique_labels):
        idx = np.where(labels_np == lab)[0]
        plt.scatter(X2[idx,0], X2[idx,1], s=15, alpha=0.7, label=lab, color=cmap(i))
    plt.title("t-SNE of CKM Decoder Parameters", fontsize=24)
    plt.legend(loc=2, prop={'size': 24})
    # out_file = os.path.join(save_path, "tsne_ckm.png")
    out_file = os.path.join(save_path, "tsne_ckm.pdf")
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"[INFO] t-SNE 图已保存: {out_file}")
    return out_file

if __name__ == "__main__":
    plot_tsne_from_decomps(
        checkpoint_dir="/media/LiuYuyao/Project/UCLAD_CNN_BMKP/checkpoints/old/",
        save_path="/media/LiuYuyao/Project/UCLAD_CNN_BMKP/tsne/",
        sample_per_class=500,
        cluster_spacing=50,
        cluster_jitter=1.0,
        tsne_perplexity=30,
        tsne_iter=1000,
        random_seed=42,
    )