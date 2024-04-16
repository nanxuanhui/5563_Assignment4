# coding= utf-8

# 导入相关库
import numpy as np
import time
import sys
import os
import re
import tensorflow as tf
# 少写两行库
import jieba

class Seq2Seq():
    '''
        具体参数:
        encoder_vec_file encoder向量文件
        decoder_vec_file decoder向量文件
        encoder_vocabulary encoder词典
        decoder_vocabulary decoder词典
        model_path 模型目录
        batch_size 批处理数
        sample_num 总样本数
        max_batches 最大迭代数
        show_epoch 保存模型步长
    '''

def __init__(self):
    # 数用于清除默认图形堆栈并重置全局默认图形
    tf.reset_default_graph()

    self.encoder_vec_file = ".vec 文件格式存在"
    self.decoder_vec_file = ".vec 文件格式存在"
    self.encoder_vocabulary = ".vocab 文件格式存在"
    self.decoder_vocabulary = ".vocab 文件格式存在"
    self.batch_size = 1
    self.max_batches = 100000
    self.show_epoch = 100
    self.model_path = 'model文件'

