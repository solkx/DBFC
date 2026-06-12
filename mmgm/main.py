import argparse
import torch
from torch.utils.data import DataLoader
import transformers

import data_loader
import utils
from mmgm import *
from tqdm import tqdm

class Trainer(object):
    def __init__(self, model):
        self.model = model
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': list(set(self.model.parameters())),
             'lr': config.learning_rate,
             'weight_decay': config.weight_decay},
        ]

        self.optimizer = transformers.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                      num_warmup_steps=config.warm_factor * updates_total,
                                                                      num_training_steps=updates_total)

    def train(self, epoch, data_loader):
        self.model.train()
        loss_list = []
        pbar = tqdm(data_loader, desc="Train: ")
        for data_batch in pbar:
            data_batch = [data.cuda() for data in data_batch[:4]]

            bert_inputs, vision_inputs = data_batch

            outputs = self.model(bert_inputs, vision_inputs)

            loss = outputs["loss"]
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_list.append(loss.cpu().item())

            self.scheduler.step()
            
            pbar.set_postfix(
                {"Loss": sum(loss_list) / len(loss_list)}
            )
            
        return sum(loss_list) / len(loss_list)

    def eval(self, epoch, data_loader, is_test=False):
        self.model.eval()
        with torch.no_grad():
            for i, data_batch in enumerate(data_loader):
                data_batch = [data.cuda() for data in data_batch[:4]]

                bert_inputs, vision_inputs = data_batch

                outputs = self.model(bert_inputs, vision_inputs)

                sim = outputs['sim']

        title = "EVAL" if not is_test else "TEST"
        logger.info(f"epoch {epoch} is sim: {round(sim, 4)}")
 
        return sim

    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./model_vae_1.0_17_xilidu.pt')
    parser.add_argument('--predict_path', type=str, default='./output.json')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset_name', type=str, default="15")

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--bert_name', type=str, default='../bert-base-uncased')
    parser.add_argument('--bert_learning_rate', type=float, default=1e-5)
    parser.add_argument('--warm_factor', type=float, default=0.1)
    parser.add_argument('--text_len', type=int, default=128)
    parser.add_argument('--image_len', type=int, default=197)
    parser.add_argument('--latent_len', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=256)

    parser.add_argument('--seed', type=int, default=3306)
    parser.add_argument('--miss_rate', type=float, default=1.0)
    parser.add_argument('--gen_type', type=str, default="vae")

    config = parser.parse_args()

    logger = utils.get_logger(config.dataset_name)
    logger.info(config)
    config.logger = logger

    if torch.cuda.is_available():
        torch.cuda.set_device(config.device)

    logger.info("Loading Data")
    datasets = data_loader.load_data_bert(config)

    train_loader, dev_loader = (
        DataLoader(dataset=dataset,
                   batch_size=config.batch_size,
                   collate_fn=data_loader.collate_fn,
                   shuffle=i == 0,
                   num_workers=4,
                   drop_last=False)
        for i, dataset in enumerate(datasets)
    )

    updates_total = len(datasets[0]) // config.batch_size * config.epochs

    model = VAE(config)

    model = model.cuda()

    trainer = Trainer(model)

    max_sim = 0
    for i in range(config.epochs):
        logger.info("Epoch: {}".format(i))
        trainer.train(i, train_loader)
        sim = trainer.eval(i, dev_loader)
        if max_sim < sim:
            max_sim = sim
            trainer.save(config.save_path)
    logger.info("Best DEV SIM: {:3.4f}".format(max_sim))
    trainer.load(config.save_path)
    trainer.eval("Final", dev_loader)
