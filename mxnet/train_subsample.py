#coding=utf8
# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme 
# pylint: disable=superfluous-parens, no-member, invalid-name
import os
import re
import gc
import time
import pprint
import pickle
import hashlib
from math import log
import sys, datetime, math, random
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
from io import BytesIO
from collections import namedtuple
from sklearn import metrics
import logging

bit = 22
mask = (1<<bit)-1
num_vocab = mask + 1
num_embed = 8
input_name = 'x'
output_name = 'y'
batch_size = 3000
CATE_FEA_LEN = 14
DENSE_FEA_LEN = 27
FEA_LEN = CATE_FEA_LEN + DENSE_FEA_LEN
PADDING_LEN = FEA_LEN + 1
eps = 0.00000001
SIZE_ZOOM=10000
SAMPLE_RATE=0.5

#与mlp中java代码一致
def signN(s):
    return int(hashlib.md5(s).hexdigest()[0:7], 16) & ((1<<bit)-1)

def logloss(ys, ps, neg_ratio=1.):
    loss_sum=0
    for i in range(len(ys)) :
        y = ys[i]
        y1 = ps[i]
        y1 = y1*neg_ratio / (1-y1+neg_ratio*y1) 
        if y1 >= 1. :
            y1 = 0.999
        y0 = 1-y1
        try:
            loss_sum += -(y*log(y1) + (1-y)*log(y0))
        except:
            print  "log error:"+str(y1)+" "+str(y0)+" "+str(y)
    return loss_sum/len(ys)

def mae(ys, ps):
    s = 0
    n = 0
    for i in range(len(ys)) :
        s += math.fabs(ys[i] - ps[i])
        n += 1
    return s/n

def load_np_data(dt_file, is_test):
    fp = open(dt_file)
    x=[]
    lcnt=0 #行数
    for line in fp :
        lcnt += 1
        #if (lcnt % 1000000) == 0 :
        #    print >> sys.stderr, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), "Read %d lines !"%(lcnt)

        parts = line.strip().split("\t")
        parts = [float(a) for a in parts]
        if len(parts) != PADDING_LEN :
            continue
        label = int(parts[0])
        #训练数据负采样
        if (not is_test) and label == 0 and random.random() >= SAMPLE_RATE:
            continue
        #测试数据全采样
        if is_test and random.random() >= SAMPLE_RATE:
            continue
        x.append(np.array(parts))
    x = np.array(x)
    fp.close()
    return x

def load_np_data_bin(txt_file, is_test):
    file_np_bin = txt_file + ".npy"
    #test文件不保存.npy
    if is_test:
        return load_np_data(txt_file, is_test)
    if not os.path.exists(file_np_bin):
        print "npy format file doesn't exist"
        x = load_np_data(txt_file, is_test)
        f = file(file_np_bin, "wb")
        print "save as npy format", file_np_bin
        np.save(f, x) #将x保存在npy文件中
        f.close()
        return x
    else :
        print "there is npy format file, load it"
        f = file(file_np_bin, "rb")
        x = np.load(f)
        f.close()
        return x

def preprocess(np_data):
    np_mean = np.mean(np_data[:,CATE_FEA_LEN+1:], axis=0)
    np_var = np.var(np_data[:,CATE_FEA_LEN+1:], axis=0)
    #标准化
    trans_dense_fea = (np_data[:,CATE_FEA_LEN+1:] - np_mean) / (np.power(np_var, 0.5) + eps)
    #归一化
    trans_dense_fea = 1 / (1 + np.exp(-2*trans_dense_fea))#sigmoid
    trans_dense_fea = 2 * trans_dense_fea - 1#tanh
    pre = np_data[:,:CATE_FEA_LEN+1]
    res = np.concatenate((pre, trans_dense_fea), axis=1)
    return np_mean, np_var, res

#更新normalize文件
def save_preprocess(np_data, prefix):
    np_mean, np_var, np_data = preprocess(np_data)
    new_size = np_data.shape[0] / float(SIZE_ZOOM)
    outfile = prefix + '-normalize.txt'
    #读文件
    try:
        fin = open(outfile, 'r')
        old_size = float(fin.readline().strip())
        old_mean_list = fin.readline().strip().split("\t")
        old_mean_list = [float(a) for a in old_mean_list]
        old_var_list = fin.readline().strip().split("\t")
        old_var_list = [float(a) for a in old_var_list]
        fin.close()
    except:
        old_size = 0.0
        old_mean_list = [0.0] * len(list(np_mean))
        old_var_list = [0.0] * len(list(np_var))
    #合并
    new_mean_list = list(np_mean)
    new_var_list = list(np_var)
    assert len(old_mean_list)==len(new_mean_list), "mean len not equal"
    mean_list = []
    var_list = []
    for i in range(len(old_mean_list)):
        mean = (new_size * new_mean_list[i] + old_size * old_mean_list[i]) / (new_size + old_size)
        mean_list.append(mean)
        var = ((new_size - 1) * new_var_list[i] + (old_size - 1) * old_var_list[i]) / (new_size + old_size - 2)
        var_list.append(var)
    mean_str = "\t".join([str(a) for a in mean_list])
    var_str = "\t".join([str(a) for a in var_list])
    #覆盖写文件
    fout = open(outfile, 'w')
    fout.write(str(new_size + old_size)+"\n")
    fout.write(mean_str+"\n")
    fout.write(var_str+"\n")
    fout.close()
    return np_mean, np_var, np_data

def get_net(num_vocab, num_embed, dropout=0.0):
    cx = mx.symbol.Variable('cx')
    dx = mx.symbol.Variable('dx')
    y = mx.symbol.Variable('y')

    embed = mx.symbol.Embedding(data = cx, input_dim = num_vocab, output_dim = num_embed, name = "cx_embed")
    #print  embed.infer_shape(cx=(1000,))
    embed_flat = mx.symbol.Flatten(embed, name = "cx_embed_flatten")
    net = mx.symbol.Concat(embed_flat, dx, name='x')
    net = mx.symbol.FullyConnected(data = net, num_hidden = 256, name = "fc1")
    net = mx.symbol.Activation(data = net, act_type="relu", name="relu1")
    #net = mx.symbol.Dropout(data=net, p=0.2)
    net = mx.symbol.FullyConnected(data = net, num_hidden = 64, name = "fc2")
    net = mx.symbol.Activation(data = net, act_type="relu", name="relu2")
    #net = mx.symbol.Dropout(data=net, p=0.2)
    net = mx.symbol.FullyConnected(data = net, num_hidden = 1, name = "out")
    net = mx.symbol.LogisticRegressionOutput(data = net, label = y, name="sigmoid")
    print net.infer_shape(cx=(1000,CATE_FEA_LEN), dx=(1000,DENSE_FEA_LEN))
    return net

def rectify_pctr(pctr):
    return pctr / (pctr + (1 - pctr) / SAMPLE_RATE)

def evaluate(mod, np_data):
    mean, var, np_data = preprocess(np_data)
    #res = mod.score(data_iter, mx.metric.Accuracy(), num_batch=100)
    data_iter = mx.io.NDArrayIter(data={'cx':np_data[:,1:CATE_FEA_LEN+1].astype(int), 'dx':np_data[:,CATE_FEA_LEN+1:]}, label={'y':np_data[:,0]}, batch_size=batch_size, shuffle=False)
    ys_eval=[]
    ps_eval=[]
    for pred, i_batch, batch in mod.iter_predict(data_iter):
        #print pred, i_batch, batch.data, batch.label
        np_pred = pred[0].asnumpy()
        np_label = batch.label[0].asnumpy()
        for i in range(np_pred.shape[0]):
            if np_pred.shape[1] == 2:#softmax
                p = np_pred[i][1]
            else:#sigmoid
                p = np_pred[i]
            if SAMPLE_RATE < 1.0:
                p = rectify_pctr(p)
            y = np_label[i]
            ps_eval.append(p)
            ys_eval.append(y)
    fpr, tpr, thresholds = metrics.roc_curve(ys_eval, ps_eval, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    loss= logloss(ys_eval, ps_eval, 1.0)
    m = mae(ys_eval, ps_eval)
    return auc, loss, m


def train(net, np_data, prefix, index, epoch):
    mean, var, np_data = save_preprocess(np_data, prefix)
    np.random.shuffle(np_data)
    #data_train = mx.io.NDArrayIter(data=[np_data[:,1:6],np_data[:,6:]], label=[np_data[:,0]], batch_size=batch_size, shuffle=False, label_name='y')
    data_train = mx.io.NDArrayIter(data={'cx':np_data[:,1:CATE_FEA_LEN+1].astype(int), 'dx':np_data[:,CATE_FEA_LEN+1:]}, label={'y':np_data[:,0]}, batch_size=batch_size, shuffle=False)
    if index > 0:
        print "loading... %s-%d"%(prefix, index)
        mod = mx.mod.Module.load(prefix, index, load_optimizer_states=True, data_names=['cx','dx'], label_names=[output_name], context=mx.gpu())
    else:
        print "initing.. %s-%d"%(prefix, index)
        #mod = mx.mod.Module(net, context=mx.gpu(), data_names=['_0_data','_1_data'], label_names=[output_name])# create a module by given a Symbol
        mod = mx.mod.Module(net, context=mx.gpu(), data_names=['cx','dx'], label_names=[output_name])# create a module by given a Symbol
    
    print "training..."
    mod.fit(data_train,
            optimizer='adadelta',
            optimizer_params={'learning_rate':0.001},
            initializer=mx.initializer.Uniform(),
            num_epoch=epoch)
    print "saving..."
    mod.save_checkpoint(prefix, index + 1, save_optimizer_states=True)
    return mod

def train_dir(data_dir, prefix, start, test_file, file_num_limit=0):
    out_loop_num = 1
    epoch = 3
    os.system('rm -f ' + prefix + '-normalize.txt')
    net = get_net(num_vocab, num_embed)
    data_list = os.listdir(data_dir)
    tmp_list = []
    for f in data_list:
        index = f.find(".npy")
        if index > -1:
            tmp_list.append(f[:index])
        else:
            tmp_list.append(f)
    data_list = [a for a in list(set(tmp_list)) if a < test_file]
    data_list.sort()
    #加载测试数据
    print "loading test data:" + test_file
    data_eval = load_np_data_bin(data_dir + "/" + test_file, True)
    if file_num_limit != 0 and file_num_limit < len(data_list):
        data_list = data_list[:file_num_limit]
    print "training files:" + ",".join(data_list)
    for i in range(out_loop_num):
        for i,file_name in enumerate(data_list):
            print time.ctime()," training file: ",file_name
            np_data = load_np_data_bin(data_dir + "/" + file_name, False)
            mod = train(net , np_data, prefix, start + i, epoch)
            print "eval: ",evaluate(mod, data_eval)
            del np_data
            gc.collect()
            print "finish %d:%s"%(start + i, file_name)
        start += len(data_list)
    return start


if __name__ == '__main__':
    l = train_dir('/data/xql/dense_subsample/','21days_3epoch_dropout/checkpoint', 0, 'dnn_dense_data_20170620')

    #net = get_net(num_vocab, num_embed)
    #np_data = load_np_data_bin('/data/xql/dense/dnn_dense_data_20170520')
    #prefix = 'checkpoint'
    #mod = train(net, np_data, prefix, 0)
    #data_eval = load_np_data_bin("/data/xql/dense/dnn_dense_data_20170523_test")
    #print evaluate(mod, data_eval)

