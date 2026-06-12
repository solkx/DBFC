import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import prettytable as pt
from transformers import AutoTokenizer
import os
import utils
import requests

os.environ["TOKENIZERS_PARALLELISM"] = "false"

dis2idx = np.zeros((1000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


class Vocabulary(object):
    PAD = '<pad>'
    UNK = '<unk>'
    SUC = '<suc>'

    def __init__(self):
        self.label2id = {self.PAD: 0, self.SUC: 1}
        self.id2label = {0: self.PAD, 1: self.SUC}

    def add_label(self, label):
        label = label.lower()
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label

        assert label == self.id2label[self.label2id[label]]

    def __len__(self):
        return len(self.token2id)

    def label_to_id(self, label):
        label = label.lower()
        return self.label2id[label]

    def id_to_label(self, i):
        return self.id2label[i]

def pad_and_stack_tensors(tensor_group):
    # 找到这一组张量的 x_max
    x_max = max(tensor.shape[0] for tensor in tensor_group)
    
    # 创建一个列表，用于存储填充后的张量
    padded_tensors = []
    
    for tensor in tensor_group:
        x = tensor.shape[0]
        
        # 如果当前张量的 x 小于 x_max，进行填充
        if x < x_max:
            # 使用torch.nn.functional.pad在第0维进行填充
            padding = (0, 0, 0, 0, 0, 0, 0, x_max - x)  # 只在第0维填充 (x_max - x)
            padded_tensor = torch.nn.functional.pad(tensor, padding, mode='constant', value=0)
        else:
            padded_tensor = tensor  # 如果已经是 x_max 的大小，无需填充
        
        padded_tensors.append(padded_tensor)
    
    # 按第一个维度进行堆叠，形成 (8, x_max, 3, 1920, 1080) 的张量
    stacked_tensor = torch.stack(padded_tensors)
    
    return stacked_tensor

def collate_fn(data):
    bert_inputs, vision_inputs, pieces2word, sent_length, image_id = map(list, zip(*data))

    bert_inputs = torch.stack([torch.cat([seq, torch.zeros(128-seq.shape[0])], dim=0) for seq in bert_inputs], dim=0).long()
    vision_inputs = pad_and_stack_tensors(vision_inputs)

    return bert_inputs, vision_inputs, pieces2word, sent_length, image_id


class RelationDataset(Dataset):
    def __init__(self, bert_inputs, vision_inputs, pieces2word, sent_length, image_id):
        self.bert_inputs = bert_inputs
        self.vision_inputs = vision_inputs
        self.pieces2word = pieces2word
        self.sent_length = sent_length
        self.image_id = image_id

    def __getitem__(self, item):
        return torch.LongTensor(self.bert_inputs[item]), \
               torch.FloatTensor(self.vision_inputs[item]), \
               torch.LongTensor(self.pieces2word[item]), \
               self.sent_length[item], \
               self.image_id[item]

    def __len__(self):
        return len(self.bert_inputs)

import cv2
from tqdm import tqdm
def process_bert(data, tokenizer, config):

    bert_inputs = []
    vision_inputs = []
    pieces2word = []
    sent_length = []
    image_id = []
    max_len = 0
    for instance in tqdm(data):
        if len(instance['tokens']) == 0:
            continue
        img = instance["img"]
        miss_type = instance["miss_type"]
        if miss_type == 0:
            img_path = f"../t2i/twitter{config.dataset_name}_data/twitter20{config.dataset_name}_images/{img}.jpg"
        else:
            img_path = f"../t2i/twitter{config.dataset_name}_data/gen_img/{img}.png"
        try:
            image = cv2.resize(cv2.imread(img_path), (224, 224), interpolation=cv2.INTER_LANCZOS4)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        except:
            image_tensor = torch.zeros((3, 224, 224)).float()

        tokens = [tokenizer.tokenize(word) for word in instance['tokens']]
        pieces = [piece for pieces in tokens for piece in pieces]
        _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
        _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])
        if len(_bert_inputs) > max_len:
            max_len = len(_bert_inputs)
        length = {i: len(l) for i, l in enumerate(tokens)}
        sent_length.append(length)
        length = len(instance['tokens'])
        _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=np.bool)

        if tokenizer is not None:
            start = 0
            for i, pieces in enumerate(tokens):
                if len(pieces) == 0:
                    continue
                pieces = list(range(start, start + len(pieces)))
                _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                start += len(pieces)

        image_id.append(img)
        vision_inputs.append(image_tensor)
        bert_inputs.append(_bert_inputs)
        pieces2word.append(_pieces2word)
    print(max_len)
    return bert_inputs, vision_inputs, pieces2word, sent_length, image_id



def txt2json(data):
    findata = []
    temp = []
    sent = []
    ents = {}
    img = ""
    for line in data:
        line = line.strip()
        if not line:
            if temp:
                ents[" ".join(temp[1:])] = temp[0]
                temp = []
            findata.append({
                "tokens": sent,
                "entity": ents,
                "img": img
            })
            sent = []
            ents = {}
            continue
        if "IMGID" in line:
            img = line.split(":")[-1]
            continue
        word = line.split("	")[0]
        label = line.split("	")[-1]
        if "-" in label:
            ent_t = label.split("-")[-1]
            label = label.split("-")[0]
            if ent_t == "OTHER":
                ent_t = "MISC"
        else:
            ent_t = ""
        sent.append(word)
        if label == "B":
            if temp:
                ents[" ".join(temp[1:])] = temp[0]
                temp = []
            temp.extend([ent_t, word])
        elif label == "I":
            if temp:
                temp.append(word)
        else:
            if temp:
                ents[" ".join(temp[1:])] = temp[0]
                temp = []
    if temp:
        ents[" ".join(temp[1:])] = temp[0]
        temp = []
    findata.append({
        "tokens": sent,
        "entity": ents,
        "img": img
    })
    return findata

def missing(data, rate=0.2):
    print(f"missing rate: {rate}")
    m_l = int(len(data)*rate)
    for ind, item in enumerate(data):
        if ind >= m_l:
            item["miss_type"] = 0
        else:
            item["miss_type"] = 1
    return data

def load_data_bert(config):

    with open(f'../t2i/twitter{config.dataset_name}_data/twitter20{config.dataset_name}/train.txt', 'r', encoding='utf-8') as f:
        train_data = missing(txt2json(f.read().split("\n")), config.miss_rate)
    with open(f'../t2i/twitter{config.dataset_name}_data/twitter20{config.dataset_name}/test.txt', 'r', encoding='utf-8') as f:
        test_data = missing(txt2json(f.read().split("\n")), config.miss_rate)
    tokenizer = AutoTokenizer.from_pretrained("../bert-base-uncased", cache_dir="./cache/")

    vocab = Vocabulary()
    config.label_num = len(vocab.label2id)
    config.vocab = vocab
    train_dataset = RelationDataset(*process_bert(train_data, tokenizer, config))
    dev_dataset = RelationDataset(*process_bert(test_data, tokenizer, config))
    return train_dataset, dev_dataset

