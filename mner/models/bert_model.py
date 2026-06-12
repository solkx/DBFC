import torch
import os
from torch import nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import ViTModel
from .modeling_bert import BertModel
from transformers.modeling_outputs import TokenClassifierOutput
import math

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

class Modal(nn.Module):
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
    
    def img_gen(self, text_feat):
        t_mu, t_logvar = self.text_encoder(text_feat, current_len=self.config.text_len)
        z_text = self.reparameterize(t_mu, t_logvar)
        
        # 正向跨模态解码
        recon_image = self.image_decoder(z_text, target_len=self.config.image_len) # bs * l2 * 768
        return recon_image
        
    def forward(self, text_feat, image_feat):
        text_feat = bert(text_feat, attention_mask=text_feat.ne(0).float())[0] # bs * l1 * 768
        image_feat = vit(image_feat)[0] # bs * l2 * 768
        # 正向编码过程
        t_mu, t_logvar = self.text_encoder(text_feat, current_len=self.config.text_len)
        z_text = self.reparameterize(t_mu, t_logvar)
        
        # 正向跨模态解码
        recon_image = self.image_decoder(z_text, target_len=self.config.image_len) # bs * l2 * 768
        # 正向一致性损失
        image_loss = (F.mse_loss(recon_image, image_feat, reduction='none')).mean()
        # 正向AVE损失
        kl_text = -0.5 * (1 + t_logvar - t_mu.pow(2) - t_logvar.exp()).mean()
        
        # 反向编码
        i_mu, i_logvar = self.image_encoder(recon_image, current_len=self.config.image_len)
        z_image = self.reparameterize(i_mu, i_logvar)

        # 反向解码
        recon_text = self.text_decoder(z_image, target_len=self.config.text_len) # bs * l1 * 768

        text_loss = (F.mse_loss(recon_text, text_feat, reduction='none') * text_feat.ne(0).float()).mean()
        kl_image = -0.5 * (1 + i_logvar - i_mu.pow(2) - i_logvar.exp()).mean()

        loss = text_loss + image_loss + kl_text + kl_image

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
        
class NERModel(nn.Module):
    def __init__(self, label_list, args):
        """
        label_list: the list of target labels
        args: argparse
        """
        super(NERModel, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.bert_name)  # get the pre-trained BERT model for the text
        self.vit = ViTModel.from_pretrained("../vit-base-patch16-224").cuda()

        self.bert_config = self.bert.config
        self.image2token_emb = 1024

        self.num_labels = len(label_list)  # the number of target labels
        self.crf = CRF(self.num_labels, batch_first=True)
        if args.gen_missing != "base":
            self.fc = nn.Linear(1000, self.num_labels)
        else:
            self.fc = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.4)

        self.linear_extend_pic = nn.Linear(self.bert.config.hidden_size, self.args.max_seq*self.image2token_emb)

        # the attention mechanism for fine-grained features
        self.linear_q_fine = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.linear_k_fine = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.linear_v_fine = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)


        # self.combine_linear = nn.Linear(self.bert.config.hidden_size, 1000)
        
        self.combine_linear = nn.Linear(self.bert.config.hidden_size + self.image2token_emb, 1000)

        self.generator = Modal(args)
        self.dla = DynamicLengthAdapter(args.image_len, 49, self.bert.config.hidden_size)
        for param in self.generator.parameters():
            param.requires_grad = False
        for param in self.dla.parameters():
            param.requires_grad = False
            
    def recover_order(self, missing_indexes, output_missing, output_present):
        final_output =[]
        miss_num, present_num = 0, 0
        for m in missing_indexes:
            if m == 1:
                final_output.append(output_missing[miss_num])
                miss_num += 1
            else:
                final_output.append(output_present[present_num])
                present_num += 1
        return final_output
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, images=None, missing_indexs=None):

        mask_missing = (missing_indexs == 1).squeeze() 
        mask_present = (missing_indexs == 0).squeeze() 
        losses, logits_list = [], []
        if int(mask_present.sum().detach().cpu().numpy()) > 0:
            output_text = self.bert(input_ids[mask_present], attention_mask[mask_present])
            hidden_text = output_text['last_hidden_state']
            pic_ori = self.vit(images[mask_present])[0]
            pic_ori_ = torch.sum(pic_ori,dim=1)

            hidden_k_text = self.linear_k_fine(hidden_text)
            hidden_v_text = self.linear_v_fine(hidden_text)
            pic_q_origin = self.linear_q_fine(pic_ori)
            pic_original = torch.sum(torch.tanh(self.att(pic_q_origin, hidden_k_text, hidden_v_text)), dim=1)

            pic_ori_final = (pic_original+pic_ori_)

            pic_ori = torch.tanh(self.linear_extend_pic(pic_ori_final).reshape(-1, self.args.max_seq, self.image2token_emb))
            emissions = self.fc(torch.relu(self.combine_linear(torch.cat([hidden_text, pic_ori], dim=-1))))
            # classification
            logits = self.crf.decode(emissions, attention_mask[mask_present].byte())
            loss = None
            if labels[mask_present] is not None:
                loss = -1 * self.crf(emissions, labels[mask_present], mask=attention_mask[mask_present].byte(), reduction='sum')
            losses.append(loss)
            logits_list.append(logits)

        if int(mask_missing.sum().detach().cpu().numpy()) > 0:
            if self.args.gen_missing == "mmgm":
                output_text = self.bert(input_ids[mask_missing], attention_mask[mask_missing])
                hidden_text = output_text['last_hidden_state']
                h_img = self.generator.img_gen(hidden_text)
                pic_ori = self.dla(h_img.permute(0, 2, 1)).permute(0, 2, 1)
                pic_ori_ = torch.sum(pic_ori,dim=1)
                hidden_k_text = self.linear_k_fine(hidden_text)
                hidden_v_text = self.linear_v_fine(hidden_text)
                pic_q_origin = self.linear_q_fine(pic_ori)
                pic_original = torch.sum(torch.tanh(self.att(pic_q_origin, hidden_k_text, hidden_v_text)), dim=1)
                
                pic_ori_final = (pic_original+pic_ori_)

                pic_ori = torch.tanh(self.linear_extend_pic(pic_ori_final).reshape(-1, self.args.max_seq, self.image2token_emb))
                emissions = self.fc(torch.relu(self.combine_linear(torch.cat([hidden_text, pic_ori], dim=-1))))
                # classification
                logits = self.crf.decode(emissions, attention_mask[mask_missing].byte())
                loss = None
                if labels[mask_missing] is not None:
                    loss = -1 * self.crf(emissions, labels[mask_missing], mask=attention_mask[mask_missing].byte(), reduction='sum')
                losses.append(loss)
                logits_list.append(logits)
            elif self.args.gen_missing == "llm":
                output_text = self.bert(input_ids[mask_missing], attention_mask[mask_missing])
                hidden_text = output_text['last_hidden_state']
                pic_ori = self.vit(images[mask_missing])[0]
                pic_ori_ = torch.sum(pic_ori,dim=1)
                hidden_k_text = self.linear_k_fine(hidden_text)
                hidden_v_text = self.linear_v_fine(hidden_text)
                pic_q_origin = self.linear_q_fine(pic_ori)
                pic_original = torch.sum(torch.tanh(self.att(pic_q_origin, hidden_k_text, hidden_v_text)), dim=1)
                
                pic_ori_final = (pic_original+pic_ori_)

                pic_ori = torch.tanh(self.linear_extend_pic(pic_ori_final).reshape(-1, self.args.max_seq, self.image2token_emb))
                emissions = self.fc(torch.relu(self.combine_linear(torch.cat([hidden_text, pic_ori], dim=-1))))
                # classification
                logits = self.crf.decode(emissions, attention_mask[mask_missing].byte())
                loss = None
                if labels[mask_missing] is not None:
                    loss = -1 * self.crf(emissions, labels[mask_missing], mask=attention_mask[mask_missing].byte(), reduction='sum')
                losses.append(loss)
                logits_list.append(logits)
            else:
                output_text = self.bert(input_ids[mask_missing], attention_mask[mask_missing])
                hidden_text = output_text['last_hidden_state']
                emissions = self.fc(hidden_text)
                # classification
                logits = self.crf.decode(emissions, attention_mask[mask_missing].byte())
                loss = None
                if labels[mask_missing] is not None:
                    loss = -1 * self.crf(emissions, labels[mask_missing], mask=attention_mask[mask_missing].byte(), reduction='sum')
                losses.append(loss)
                logits_list.append(logits)

        if mask_missing.sum() > 0 and mask_present.sum() > 0:
            logits_list = self.recover_order(missing_indexs, logits_list[1], logits_list[0])
        else:
            logits_list = logits_list[0]

        return TokenClassifierOutput(
            loss=torch.stack(losses).mean(),
            logits=logits_list
        )

    # the attention mechanism
    def att(self, query, key, value):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)  # (5,50)
        att_map = F.softmax(scores, dim=-1)

        return torch.matmul(att_map, value)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(sum=0.0, std=0.05)



