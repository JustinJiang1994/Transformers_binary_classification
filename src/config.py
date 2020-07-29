# -*- coding: utf-8 -*-

"""
Created on 2020-07-29 09:03
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com

配置模型、路径、与训练相关参数
"""

class Config(object):
    def __init__(self):
        self.config_dict = {
            "data_path": {
                "trainingSet_path": "../data/sentiment/sentiment.train.data",
                "valSet_path": "../data/sentiment/sentiment.valid.data",
                "testingSet_path": "../data/sentiment/sentiment.test.data"
            },

            "BERT_path": {
                "file_path": '../chinese_L-12_H-768_A-12/',
                "config_path": '../chinese_L-12_H-768_A-12/bert_config.json',
                "vocab_path": '../chinese_L-12_H-768_A-12/vocab.txt',
            },

            "training_rule": {
                "max_length": 300, # 输入序列长度，别超过512
                "hidden_dropout_prob": 0.3,
                "num_labels": 2,
                "learning_rate": 1e-5,
                "weight_decay": 1e-2,
                "batch_size": 16
            },

            "result": {
                "model_save_path": '../result/bert_clf.pkl'
            }
        }

    def get(self, section, name):
        return self.config_dict[section][name]