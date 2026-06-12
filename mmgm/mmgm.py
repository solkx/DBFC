import torch
import torch.nn as nn
from transformers import BertModel, ViTModel
import torch.nn.functional as F
from utils import pro, image_pro

bert = BertModel.from_pretrained("../bert-base-uncased").cuda()
vit = ViTModel.from_pretrained("../vit-base-patch16-224").cuda()

class DynamicLengthAdapter(nn.Module):
    """动态长度适配模块"""
    def __init__(self, src_len, tgt_len, feat_dim=768):
        super().__init__()
        self.scale_factor = tgt_len / src_len
        
        # 自动选择上采样或下采样模式
        if self.scale_factor >= 1:
            self.adapter = nn.Sequential(
                nn.Upsample(scale_factor=self.scale_factor, mode='nearest'),
                nn.Conv1d(feat_dim, feat_dim, 3, padding=1)
            )
        else:
            self.adapter = nn.Sequential(
                nn.AvgPool1d(kernel_size=int(1/self.scale_factor)),
                nn.Conv1d(feat_dim, feat_dim, 3, padding=1)
            )
    
    def forward(self, x):
        # 输入形状: [B, C, L]
        return self.adapter(x)

class UniversalEncoder(nn.Module):
    """通用编码器"""
    def __init__(self, config, base_len=128, feat_dim=768):
        super().__init__()
        self.base_len = base_len  # 基准长度（文本128/图像197）
        self.config = config
        
        # 第一阶段：特征提取
        self.conv1 = nn.Conv1d(feat_dim, 512, 3, padding=1)
        self.ln1 = nn.LayerNorm(512)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(512, config.latent_dim, 3, padding=1)
        self.ln2 = nn.LayerNorm(config.latent_dim)

        
        # 第二阶段：动态长度对齐
        self.length_adapter = DynamicLengthAdapter(base_len, config.latent_len, config.latent_dim)  # 统一压缩到32
        
        # 潜在空间映射
        self.to_latent = nn.Linear(config.latent_dim*config.latent_len, 2*config.latent_dim)

    def forward(self, x, current_len):
        bs = x.shape[0]
        # 特征提取
        x = self.conv1(x.permute(0, 2, 1))  # [B, 256, L]
        x = self.relu(self.ln1(x.permute(0, 2, 1)))  # [B, 256, L]
        x = self.conv2(x.permute(0, 2, 1))  # [B, 256, L]
        x = self.ln2(x.permute(0, 2, 1))  # [B, 256, L]
        
        # 统一压缩到32长度
        x = self.length_adapter(x.permute(0, 2, 1))  # [B, 256, 32]
        
        # 潜在空间映射
        mu_logvar = self.to_latent(x.view(bs, -1))
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        
        return mu, logvar

class UniversalDecoder(nn.Module):
    """通用解码器"""
    def __init__(self, config, base_len=197, feat_dim=768):
        super().__init__()
        self.config = config
        self.base_len = base_len
        # self.latent_dim = latent_dim
        
        # 潜在空间扩展
        self.from_latent = nn.Linear(config.latent_dim, config.latent_dim*config.latent_len)
        # Rearrange('b (d l) -> b d l', d=256, l=32)
        
        # 动态长度恢复
        self.length_adapter = DynamicLengthAdapter(config.latent_len, base_len, config.latent_dim)
        
        # 特征重建
   
        self.conv1 = nn.Conv1d(config.latent_dim, 512, 3, padding=1)
        self.ln1 = nn.LayerNorm(512)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(512, feat_dim, 3, padding=1)
    

    def forward(self, z, target_len=None):
        target_len = target_len or self.base_len
        
        bs = z.shape[0]
        # 潜在空间解码
        x = self.from_latent(z).view(bs, self.config.latent_dim, -1)  # [B, 256, 32]

        x = self.length_adapter(x)
        
        # 特征重建
        x = self.conv1(x)  # [B, D, L]
        x = self.relu(self.ln1(x.permute(0, 2, 1)))  # [B, D, L]
        x = self.conv2(x.permute(0, 2, 1))  # [B, D, L]
        return x.permute(0, 2, 1)

class VAE(nn.Module):
    """动态跨模态VAE"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 文本编码器（基准长度128）
        self.text_encoder = UniversalEncoder(config, config.text_len)
        # 图像编码器（基准长度197）
        self.image_encoder = UniversalEncoder(config, config.image_len)
        
        # 共享解码器
        self.text_decoder = UniversalDecoder(config, config.text_len)
        self.image_decoder = UniversalDecoder(config, config.image_len)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def _compute_reconstruction_error_weights(self, H_orig, H_recon):
        # 1. 计算逐Token的均方误差 (MSE) -> (B, L)
        mse = torch.mean((H_orig - H_recon) ** 2, dim=-1)
        
        weights = torch.exp(mse / 0.5)

        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)
        
        return weights
    
    def forward(self, text_feat, image_feat):
        text_feat = bert(text_feat, attention_mask=text_feat.ne(0).float())[0] # bs * l1 * 768
        image_feat = vit(image_feat)[0] # bs * l2 * 768

        # 正向编码过程
        t_mu, t_logvar = self.text_encoder(text_feat, current_len=self.config.text_len)
        z_text = self.reparameterize(t_mu, t_logvar)
        
        # 正向跨模态解码
        recon_image = self.image_decoder(z_text, target_len=self.config.image_len) # bs * l2 * 768
        # 正向一致性损失
        image_loss = (F.mse_loss(recon_image, image_feat, reduction='none')).mean(-1)
        # 正向AVE损失
        kl_text = -0.5 * (1 + t_logvar - t_mu.pow(2) - t_logvar.exp()).mean(-1)
        
        # 反向编码
        i_mu, i_logvar = self.image_encoder(recon_image, current_len=self.config.image_len)
        z_image = self.reparameterize(i_mu, i_logvar)

        # 反向解码
        recon_text = self.text_decoder(z_image, target_len=self.config.text_len) # bs * l1 * 768

        text_loss = (F.mse_loss(recon_text, text_feat, reduction='none') * text_feat.ne(0).float()).mean(-1)
        kl_image = -0.5 * (1 + i_logvar - i_mu.pow(2) - i_logvar.exp()).mean(-1)
        text_weight = self._compute_reconstruction_error_weights(recon_text, text_feat)
        image_weight = self._compute_reconstruction_error_weights(recon_image, image_feat)
        loss = ((text_loss * text_weight).mean(-1) + (image_loss * image_weight).mean(-1) + kl_text + kl_image).sum()

        cos_sim_text = F.cosine_similarity(text_feat, recon_text, dim=-1).mean()
        cos_sim_img = F.cosine_similarity(image_feat, recon_image, dim=-1).mean()
        sim = float(0.5 * (cos_sim_text + cos_sim_img))

        return {
            'recon_text': recon_text,
            'recon_image': recon_image,
            'z_text': z_text,
            'z_image': z_image,
            't_mu': t_mu,
            't_logvar': t_logvar,
            'i_mu': i_mu,
            'i_logvar': i_logvar,
            "loss": loss,
            "sim": sim
        }