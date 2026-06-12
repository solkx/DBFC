import os
import argparse
import logging
import sys
sys.path.append("..")

import torch
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from models.bert_model import NERModel
from processor.dataset import NERProcessor, NERDataset
from modules.train import NERTrainer

import warnings, time
warnings.filterwarnings("ignore", category=UserWarning)
# from tensorboardX import SummaryWriter

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    'twitter15': NERModel,
    'twitter17': NERModel
}

TRAINER_CLASSES = {
    'twitter15': NERTrainer,
    'twitter17': NERTrainer
}
DATA_PROCESS = {
    'twitter15': (NERProcessor, NERDataset),
    'twitter17': (NERProcessor, NERDataset)
}

DATA_PATH = {
    'twitter15': {
                # input text data
                'train': '../t2i/twitter15_data/twitter2015/train.txt',
                'dev':  '../t2i/twitter15_data/twitter2015/val.txt',
                'test':  '../t2i/twitter15_data/twitter2015/test.txt',
    },

    'twitter17': {
                # text data
                'train': '../t2i/twitter17_data/twitter2017/train.txt',
                'dev':  '../t2i/twitter17_data/twitter2017/val.txt',
                'test':  '../t2i/twitter17_data/twitter2017/test.txt',
            },
        
}

# original image data
IMG_PATH = {
    'twitter15': '../t2i/twitter15_data/twitter2015_images',
    'twitter17': '../t2i/twitter17_data/twitter2017_images',
}



def get_logger(config):
    pathname = f"./{config.log_path}/log.txt"
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

def seed_torch(seed=3306):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8' # 或 ':4096:8'

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='twitter15', type=str, help="The name of dataset.")
    parser.add_argument('--bert_name', default='../bert-base-uncased', type=str, help="Pretrained language model path")
    parser.add_argument('--num_epochs', default=30, type=int, help="num training epochs")
    parser.add_argument('--device', default='cuda', type=str, help="cuda or cpu")
    parser.add_argument('--batch_size', default=32, type=int, help="batch size")
    parser.add_argument('--lr', default=1e-5, type=float, help="learning rate")
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--eval_begin_epoch', default=1, type=int, help="epoch to start evluate")
    parser.add_argument('--seed', default=1, type=int, help="random seed, default is 1")
    parser.add_argument('--load_path', default=None, type=str, help="Load model from load_path")
    parser.add_argument('--save_path', default=None, type=str, help="save model at save_path")
    parser.add_argument('--write_path', default=None, type=str, help="do_test=True, predictions will be write in write_path")
    parser.add_argument('--notes', default="", type=str, help="input some remarks for making save path dir.")
    parser.add_argument('--do_train', default=True, type=str2bool)
    parser.add_argument('--only_test', default=True, type=str2bool)
    parser.add_argument('--max_seq', default=128, type=int)
    parser.add_argument('--ignore_idx', default=-100, type=int)
    parser.add_argument('--sample_ratio', default=1.0, type=float, help="only for low resource.")
    parser.add_argument('--log_path', default="./log", type=str)
    parser.add_argument('--missing_rate', default=0.6, type=float)
    parser.add_argument('--gen_missing', default="mmgm", type=str, choices=['mmgm', "llm", "base"], help="the method to generate missing modality, mmgm: our proposed method, llm: directly use llm to generate missing modality, base: not completing missing modality")
    parser.add_argument('--text_len', type=int, default=128)
    parser.add_argument('--image_len', type=int, default=197)
    parser.add_argument('--latent_len', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--vae_model_path', type=str, default=None)

    args = parser.parse_args()
    
    seed_torch(args.seed)
    # Some basic settings
    args.log_path = f"{args.log_path}/{args.seed}_{args.batch_size}_{args.num_epochs}_{args.lr}/{time.strftime('%m-%d_%H-%M-%S')}"
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    args.save_path = f'{args.log_path}/ckpt.pth'

    logger = get_logger(args)
    data_path, img_path = DATA_PATH[args.dataset_name], IMG_PATH[args.dataset_name]
    model_class, Trainer = MODEL_CLASSES[args.dataset_name], TRAINER_CLASSES[args.dataset_name]
    data_process, dataset_class = DATA_PROCESS[args.dataset_name]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    print(args)

    writer=None

    processor = data_process(data_path, args.bert_name, args.missing_rate)
    train_dataset = dataset_class(processor, transform, img_path, args.max_seq, sample_ratio=args.sample_ratio, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    dev_dataset = dataset_class(processor, transform, img_path,args.max_seq, mode='dev')
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    test_dataset = dataset_class(processor, transform, img_path, args.max_seq, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    label_mapping = processor.get_label_mapping()
    label_list = list(label_mapping.keys())
    model = NERModel(label_list, args)

    vae = torch.load(args.vae_model_path)
    model_dict = model.state_dict()
    model_dict.update({f"generator.{k}":v for k,v in vae.items()})
    model.load_state_dict(model_dict)

    trainer = Trainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader, model=model,
                      label_map=label_mapping, args=args, logger=logger, writer=writer)

    if args.do_train:
        # train
        trainer.train()
        # test best model
        args.load_path = f"{args.log_path}/ckpt.pth"
        trainer.test()

    if args.only_test:
        # only do test
        args.load_path = f"{args.log_path}/ckpt.pth"
        trainer.test()

    torch.cuda.empty_cache()
    # writer.close()

if __name__ == "__main__":
    main()
