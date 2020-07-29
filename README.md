# Transformers_BERT_binary_classification

基于HuggingFacek开发的Transformers库，使用BERT构建模型完成一基于中文语料的二分类模型。



## 环境

torch：1.5.1

transformers：3.0.2

tensorflow：2.2.0（不一定需要，若要将BERT进行转版本时需要）



## 数据

使用苏神的中文评论情感二分类数据集

原始Github链接：https://github.com/bojone/bert4keras/tree/master/examples/datasets

下载链接: https://pan.baidu.com/s/1gx_Im3S8wbPpUrLqPMwR5Q  密码: 8s9n



## BERT

注意：需要将BERT进行转档至Pytorch可用的版本，若有需要pytorch版本的BERT模型请在issue留言。



## 结果

约20个epochs可以使Acc到90+%

