# -*- coding: utf-8 -*-

"""
Created on 2020-07-29 19:05
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com

使用基于HuggingFace开发的transformers库进行BERT模型的加载，并构建一个二分类模型

"""

import torch
import transformers
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification
from config import Config
from torch.utils.data import DataLoader
from transformers import AdamW
from utils import SentimentDataset, convert_text_to_ids, seq_padding


class transformers_bert_binary_classification(object):
    def __init__(self):
        self.config = Config()
        self.device_setup()

    def device_setup(self):
        """
        设备配置并加载BERT模型
        :return:
        """

        # 使用GPU，通过model.to(device)的方式使用
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        MODEL_PATH = self.config.get("BERT_path", "file_path")
        config_PATH = self.config.get("BERT_path", "config_path")
        vocab_PATH = self.config.get("BERT_path", "vocab_path")

        num_labels = self.config.get("training_rule", "num_labels")
        hidden_dropout_prob = self.config.get("training_rule", "hidden_dropout_prob")

        # 通过词典导入分词器
        self.tokenizer = transformers.BertTokenizer.from_pretrained(vocab_PATH)
        self.model_config = BertConfig.from_pretrained(config_PATH, num_labels=num_labels, hidden_dropout_prob=hidden_dropout_prob)
        self.model = BertForSequenceClassification.from_pretrained(MODEL_PATH, config=self.model_config)
        self.model.to(self.device)

    def model_setup(self):
        weight_decay = self.config.get("training_rule", "weight_decay")
        learning_rate = self.config.get("training_rule", "learning_rate")

        # 定义优化器和损失函数
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()


    def get_data(self):
        """
        读取数据
        :return:
        """
        train_set_path = self.config.get("data_path", "trainingSet_path")
        valid_set_path = self.config.get("data_path", "valSet_path")
        batch_size = self.config.get("training_rule", "batch_size")

        # 数据读入
        # 加载数据集
        sentiment_train_set = SentimentDataset(train_set_path)
        sentiment_train_loader = DataLoader(sentiment_train_set, batch_size=batch_size, shuffle=True, num_workers=2)
        sentiment_valid_set = SentimentDataset(valid_set_path)
        sentiment_valid_loader = DataLoader(sentiment_valid_set, batch_size=batch_size, shuffle=False, num_workers=2)

        return sentiment_train_loader, sentiment_valid_loader


    def train_an_epoch(self, iterator):
        self.model.train()
        self.model_setup()
        epoch_loss = 0
        epoch_acc = 0

        for i, batch in enumerate(iterator):
            label = batch["label"]
            text = batch["text"]
            input_ids, token_type_ids = convert_text_to_ids(self.tokenizer, text)
            input_ids = seq_padding(self.tokenizer, input_ids)
            token_type_ids = seq_padding(self.tokenizer, token_type_ids)
            # 标签形状为 (batch_size, 1)
            label = label.unsqueeze(1)
            # 需要 LongTensor
            input_ids, token_type_ids, label = input_ids.long(), token_type_ids.long(), label.long()
            # 梯度清零
            self.optimizer.zero_grad()
            # 迁移到GPU
            input_ids, token_type_ids, label = input_ids.to(self.device), token_type_ids.to(self.device), label.to(self.device)
            output = self.model(input_ids=input_ids, token_type_ids=token_type_ids, labels=label)
            y_pred_prob = output[1]
            y_pred_label = y_pred_prob.argmax(dim=1)
            # 计算loss
            # 这个 loss 和 output[0] 是一样的
            loss = self.criterion(y_pred_prob.view(-1, 2), label.view(-1))
            #loss = output[0]
            # 计算acc
            acc = ((y_pred_label == label.view(-1)).sum()).item()
            # 反向传播
            loss.backward()
            self.optimizer.step()
            # epoch 中的 loss 和 acc 累加
            epoch_loss += loss.item()
            epoch_acc += acc
            if i % 200 == 0:
                print("current loss:", epoch_loss / (i+1), "\t", "current acc:", epoch_acc / ((i+1)*len(label)))
        return epoch_loss / len(iterator), epoch_acc / len(iterator.dataset.dataset)

    def evaluate(self, iterator):
        self.model.eval()
        epoch_loss = 0
        epoch_acc = 0
        with torch.no_grad():
            for _, batch in enumerate(iterator):
                label = batch["label"]
                text = batch["text"]
                input_ids, token_type_ids = convert_text_to_ids(self.tokenizer, text)
                input_ids = seq_padding(self.tokenizer, input_ids)
                token_type_ids = seq_padding(self.tokenizer, token_type_ids)
                label = label.unsqueeze(1)
                input_ids, token_type_ids, label = input_ids.long(), token_type_ids.long(), label.long()
                input_ids, token_type_ids, label = input_ids.to(self.device), token_type_ids.to(self.device), label.to(self.device)
                output = self.model(input_ids=input_ids, token_type_ids=token_type_ids, labels=label)
                y_pred_label = output[1].argmax(dim=1)
                loss = output[0]
                acc = ((y_pred_label == label.view(-1)).sum()).item()
                epoch_loss += loss.item()
                epoch_acc += acc
        return epoch_loss / len(iterator), epoch_acc / len(iterator.dataset.dataset)


    def train(self, epochs):
        model_save_path = self.config.get("result", "model_save_path")
        sentiment_train_loader, sentiment_valid_loader = self.get_data()

        for i in range(epochs):
            train_loss, train_acc = self.train_an_epoch(sentiment_train_loader)
            print("train loss: ", train_loss, "\t", "train acc:", train_acc)
            valid_loss, valid_acc = self.evaluate(sentiment_valid_loader)
            print("valid loss: ", valid_loss, "\t", "valid acc:", valid_acc)
        torch.save(self.model, model_save_path)

if __name__ == '__main__':
    classifier = transformers_bert_binary_classification()
    classifier.train(1)
