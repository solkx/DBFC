import random
import os
import torch
import json
import ast
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torchvision import transforms
import logging
import json
import numpy as np
logger = logging.getLogger(__name__)

class NERProcessor(object):
    def __init__(self, data_path, bert_name, missing_rate) -> None:
        self.data_path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True)
        self.missing_rate = missing_rate

    def replace_zeros_numpy(self, m, rate):
        replace_num = int(m * rate)
        return [1] * replace_num + [0] * (m - replace_num)
        
    def load_from_file(self, mode="train", sample_ratio=1.0):
        load_file = self.data_path[mode]
        logger.info("Loading data from {}".format(load_file))

        # split text and img id
        with open(load_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            raw_words, raw_targets = [], []
            raw_word, raw_target = [], []
            imgs = []
            for ii, line in enumerate(lines):
                if line.startswith("IMGID:"):
                    img_id = line.strip().split('IMGID:')[1] + '.jpg'
                    img_id_dif = line + '.png'
                    imgs.append(img_id)
                    continue
                if line != "\n":
                    raw_word.append(line.split('\t')[0])
                    label = line.split('\t')[1][:-1]
                    if 'OTHER' in label:
                        label = label[:2] + 'MISC'
                    raw_target.append(label)
                else:
                    raw_words.append(raw_word)
                    raw_targets.append(raw_target)
                    raw_word, raw_target = [], []
            raw_words.append(raw_word)
            raw_targets.append(raw_target)
        assert len(raw_words) == len(raw_targets) == len(imgs), "{}, {}, {}".format(len(raw_words), len(raw_targets), len(imgs))


        # sample data, only for low-resource
        if sample_ratio != 1.0:
            sample_indexes = random.choices(list(range(len(raw_words))), k=int(len(raw_words)*sample_ratio))
            sample_raw_words = [raw_words[idx] for idx in sample_indexes]
            sample_raw_targets = [raw_targets[idx] for idx in sample_indexes]
            sample_imgs = [imgs[idx] for idx in sample_indexes]
            assert len(sample_raw_words) == len(sample_raw_targets) == len(sample_imgs), "{}, {}, {}".format(len(sample_raw_words), len(sample_raw_targets), len(sample_imgs))
            return {"words": sample_raw_words, "targets": sample_raw_targets, "imgs": sample_imgs}

        missing_indexs = self.replace_zeros_numpy(len(imgs), self.missing_rate)
        return {"words": raw_words, "targets": raw_targets, "imgs": imgs, "missing_indexs": missing_indexs}

    # transform labels to numbers
    def get_label_mapping(self):
        LABEL_LIST = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
        label_mapping = {label:idx for idx, label in enumerate(LABEL_LIST, 1)}
        label_mapping["PAD"] = 0
        return label_mapping

class NERDataset(Dataset):
    def __init__(self, processor, transform, img_path=None, max_seq=40, sample_ratio=1, mode='train', ignore_idx=0) -> None:
        self.processor = processor
        self.transform = transform
        self.data_dict = processor.load_from_file(mode, sample_ratio)
        self.tokenizer = processor.tokenizer
        self.label_mapping = processor.get_label_mapping()
        self.max_seq = max_seq
        self.ignore_idx = ignore_idx
        self.img_path = img_path
        self.mode = mode
        self.sample_ratio = sample_ratio

    def __len__(self):
        return len(self.data_dict['words'])

    def __getitem__(self, idx):
        # get input text, labels and two kinds of images
        word_list, label_list, img = self.data_dict['words'][idx], self.data_dict['targets'][idx], self.data_dict['imgs'][idx]

        missing_indexs = self.data_dict["missing_indexs"][idx]
        # text processing by BERT
        tokens, labels, missing_indexss = [], [], []
        missing_indexss.append(missing_indexs)
        for i, word in enumerate(word_list):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            label = label_list[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(self.label_mapping[label])
                else:
                    labels.append(self.label_mapping["X"])

        if len(tokens) >= self.max_seq - 1:
            tokens = tokens[0:(self.max_seq - 2)]
            labels = labels[0:(self.max_seq - 2)]
        encode_dict = self.tokenizer.encode_plus(tokens, max_length=self.max_seq, truncation=True, padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], encode_dict['attention_mask']
        labels = [self.label_mapping["[CLS]"]] + labels + [self.label_mapping["[SEP]"]] + [self.ignore_idx]*(self.max_seq-len(labels)-2)


        # image process
        if self.img_path is not None:
            # fine-grained image feature processing
            # if missing_indexs:
            img_path = os.path.join(self.img_path, img)
            # else:
            #     img_path = os.path.join("/".join(self.img_path.split("/")[:-1] + ["gen_img"]), img).replace(".jpg", ".png")
            try:
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
            except UnidentifiedImageError:
            #     # if the image doesn't exist, use all zero tensors for substitution
                image = torch.zeros((3, 224, 224))
            
        assert len(input_ids) == len(token_type_ids) == len(attention_mask) == len(labels)
        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(labels), image, torch.tensor(missing_indexss)
