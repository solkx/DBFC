import logging
import pickle
import time
from collections import defaultdict, deque
import torch.nn.functional as F
import torch
import numpy as np
from PIL import Image
import requests
from torchvision import transforms
import timm
import matplotlib.pyplot as plt

def vit_preprocess_pipeline(input_path, output_path, target_size=224):
    """
    直接将图像强制缩放（采样）为 224x224
    """
    try:
        # 1. 打开图片
        img = Image.open(input_path).convert('RGB')
        
        # 2. 直接强制缩放至 224x224
        # 这一步会改变长宽比，如果原图不是正方形，图像会被拉伸或压扁
        # Image.Resampling.LANCZOS 是一种高质量的采样滤镜
        img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
        
        # 3. 保存
        img_resized.save(output_path)
        print(f"✅ 完成：图片已直接采样为 224x224 并保存至 {output_path}")
        
    except Exception as e:
        print(f"❌ 出错：{e}")

from matplotlib.colors import LinearSegmentedColormap
def image_pro(tensor_196, input_path, original_img_path="./output.jpg", output_path='pred_1.jpg'):
    vit_preprocess_pipeline(input_path, original_img_path)
    img = Image.open(original_img_path).convert('RGB')
    img = img.resize((224, 224))
    img_np = np.array(img)
    
    # 2. 处理 Tensor 数据
    if isinstance(tensor_196, torch.Tensor):
        values = tensor_196.cpu().numpy()
    else:
        values = np.array(tensor_196)
    
    # 归一化到 0-1
    min_val = values.min()
    max_val = values.max()
    
    if max_val - min_val == 0:
        norm_values = np.zeros_like(values)
    else:
        norm_values = (values - min_val) / (max_val - min_val)
    
    # 3. 上采样：从 196 个点 -> 224x224 的块
    # 先变成 14x14 的网格
    grid_14 = norm_values.reshape(14, 14)
    # 使用 Kronecker 积放大 16 倍 (14*16=224)，保持方块感
    heatmap_224 = np.kron(grid_14, np.ones((16, 16)))
    
    # 4. 定义自定义颜色映射 (红 -> 黄 -> 浅绿 -> 绿)
    # 颜色定义使用 0-1 的 RGB 值
    colors = [
        (1.0, 0.0, 0.0),  # 红色 (数值最小，最深/最危险)
        (1.0, 1.0, 0.0),  # 黄色 (中间过渡)
        (0.6, 1.0, 0.6),  # 浅绿 (偏绿，较亮)
        (0.0, 0.8, 0.0)   # 绿色 (数值最大，最浅/最安全)
    ]
    
    # 创建 colormap
    # 注意：因为你的需求是“数值越小越深(红)”，而通常 cmap(0) 是低端，cmap(1) 是高端
    # 所以我们需要反转这个顺序，或者在绘图时反转数据。
    # 这里我们定义一个从 绿->红 的 cmap，然后在绘图时把数据反转 (1 - heatmap)，
    # 这样 0(最小值) 就会对应列表里的红色。
    custom_cmap = LinearSegmentedColormap.from_list("custom_red_green", colors)
    
    # 5. 绘图
    plt.figure(figsize=(8, 8))
    
    # 显示底层原图
    plt.imshow(img_np)
    
    # 叠加热力图
    # 关键点：
    # 1. heatmap_224 数据本身是 0(小) 到 1(大)。
    # 2. 我们想要 0(小) 显示红色。
    # 3. custom_cmap 是 绿->红。
    # 4. 所以我们直接传 (1 - heatmap_224) 进去，让最小值(0)变成1(红色)，最大值(1)变成0(绿色)。
    # 5. alpha=0.5 让颜色变浅，透出底下的图。
    plt.imshow(1 - heatmap_224, cmap=custom_cmap, alpha=0.5, interpolation='nearest')

    plt.axis('off')
    plt.tight_layout()
    
    # 保存
    plt.savefig(output_path, dpi=150)
    print(f"✅ 自定义热力图已保存至: {output_path}")
    plt.show()

    
def pro(q, k):
    res = []
    s = 0
    for i, j in k.items():
        dl = q[s:s+j]
        d = sum(dl) / len(dl)
        res.append(d)
        s += j
    return res

def vae_loss(
    recon,
    target,
    mu,
    logvar,
    beta=1.0,
    adversarial_loss=None,
    lambda_adv=0.1,
):
    # 重构损失（MSE）
    recon_loss = F.mse_loss(recon, target)
    
    # KL散度损失
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 总VAE损失
    total_loss = recon_loss + beta * kld_loss
    
    # 如果存在对抗损失，则添加
    if adversarial_loss is not None:
        total_loss += lambda_adv * adversarial_loss
    
    return {
        "total_loss": total_loss,
        "recon_loss": recon_loss,
        "kld_loss": kld_loss,
        "adv_loss": adversarial_loss if adversarial_loss is not None else 0.0
    }

def discriminator_loss(
    disc_real,
    disc_fake,
    loss_type="bce"
):
    if loss_type == "bce":
        real_loss = F.binary_cross_entropy(disc_real, torch.ones_like(disc_real))
        fake_loss = F.binary_cross_entropy(disc_fake, torch.zeros_like(disc_fake))
    elif loss_type == "wasserstein":
        real_loss = -torch.mean(disc_real)
        fake_loss = torch.mean(disc_fake)
    return real_loss + fake_loss

def get_logger(dataset):
    pathname = "./log/{}_{}.txt".format(dataset, time.strftime("%m-%d_%H-%M-%S"))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def save_file(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text


def convert_text_to_index(text):
    index, type = text.split("-#-")
    index = [int(x) for x in index.split("-")]
    return index, int(type)


def decode(outputs, entities, length):
    class Node:
        def __init__(self):
            self.THW = []                # [(tail, type)]
            self.NNW = defaultdict(set)   # {(head,tail): {next_index}}

    ent_r, ent_p, ent_c = 0, 0, 0
    decode_entities = []
    q = deque()
    for instance, ent_set, l in zip(outputs, entities, length):
        predicts = []
        nodes = [Node() for _ in range(l)]
        for cur in reversed(range(l)):
            heads = []
            for pre in range(cur+1):
                # THW
                if instance[cur, pre] > 1: 
                    nodes[pre].THW.append((cur, instance[cur, pre]))
                    heads.append(pre)
                # NNW
                if pre < cur and instance[pre, cur] == 1:
                    # cur node
                    for head in heads:
                        nodes[pre].NNW[(head,cur)].add(cur)
                    # post nodes
                    for head,tail in nodes[cur].NNW.keys():
                        if tail >= cur and head <= pre:
                            nodes[pre].NNW[(head,tail)].add(cur)
            # entity
            for tail,type_id in nodes[cur].THW:
                if cur == tail:
                    predicts.append(([cur], type_id))
                    continue
                q.clear()
                q.append([cur])
                while len(q) > 0:
                    chains = q.pop()
                    for idx in nodes[chains[-1]].NNW[(cur,tail)]:
                        if idx == tail:
                            predicts.append((chains + [idx], type_id))
                        else:
                            q.append(chains + [idx])
        
        predicts = set([convert_index_to_text(x[0], x[1]) for x in predicts])
        decode_entities.append([convert_text_to_index(x) for x in predicts])
        ent_r += len(ent_set)
        ent_p += len(predicts)
        ent_c += len(predicts.intersection(ent_set))
    return ent_c, ent_p, ent_r, decode_entities


def cal_f1(c, p, r):
    if r == 0 or p == 0:
        return 0, 0, 0

    r = c / r if r else 0
    p = c / p if p else 0

    if r and p:
        return 2 * p * r / (p + r), p, r
    return 0, p, r
