#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import time
import csv
import pickle
import operator
import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose/sample')
opt = parser.parse_args()
print(opt)

dataset = 'sample_train-item-views.csv'
if opt.dataset == 'diginetica':
    dataset = 'train-item-views.csv'
elif opt.dataset =='yoochoose':
    dataset = 'yoochoose-clicks.dat'

print("-- Starting @ %ss" % datetime.datetime.now())
#这段程序主要是为了把数据输入到对应的字典中
#sess_clicks{'会话1':[物品id1，物品id2,..],'会话2':[物品id1，物品id2..]..}
#sess_date{'会话1':年月日1,'会话2':年月日2....],..}
#ctr点击的总次数 curdate最后一个会话序列点击的最后时间 curid会话id
with open(dataset, "r") as f:
    if opt.dataset == 'yoochoose':
        reader = csv.DictReader(f, delimiter=',')
    else:
        reader = csv.DictReader(f, delimiter=';')
    sess_clicks = {} #sess_clicks{'会话1':[(物品id1,时间戳1)，(物品id2,时间戳2)..],'会话2':[(物品id1,时间戳1)，(物品id2,时间戳2)..]..}
    sess_date = {} #sess_date{'会话1':年月日1,'会话2':年月日2....],
    ctr = 0    #点击次数
    curid = -1  #一般是会话id
    curdate = None
    for data in reader:
        sessid = data['session_id']
        if curdate and not curid == sessid: #如果当前curid不等于会话id，说明第一个会话循环已经结束，将第一个会话的截至时间存储下来，这一步是为了往sess_date里添东西
            date = ''
            if opt.dataset == 'yoochoose':
                date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
            else:
                date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            sess_date[curid] = date   #将年月日存储到字典sess_date里面
        curid = sessid
        if opt.dataset == 'yoochoose':
            item = data['item_id']
        else:
            item = data['item_id'], int(data['timeframe'])   #item('物品id',时间戳)
        curdate = ''
        if opt.dataset == 'yoochoose':
            curdate = data['timestamp']
        else:
            curdate = data['eventdate']

        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]           #把item('物品id',时间戳)往sess_clicks{}里面添加
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
    date = ''
    if opt.dataset == 'yoochoose':
        date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
    else:
        date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
        for i in list(sess_clicks):
            sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))    #sorted_clicks与sess_clicks的区别就是没有会话
            sess_clicks[i] = [c[0] for c in sorted_clicks]        #
    sess_date[curid] = date
print("-- Reading data @ %ss" % datetime.datetime.now())

# Filter out length 1 sessions        删掉一个会话中只点击过一次物品的会话，此时的sess_clicks{'会话1':[物品id1，物品id2,..],'会话2':[物品id1，物品id2..]..}
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]

# Count number of times each item appears      统计点击过的物品在所有的会话中出现的次数  iid_counts{物品id:物品出现次数}
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))   #按出现次数对item进行排序   sorted_counts[(物品id1,1),(物品id2,1),...]

length = len(sess_clicks)

#整段程序把物品点击次数小于5且筛选了次数后的会话物品个数小于2的会话删掉
for s in list(sess_clicks):
    # 当前这个会话中被点击的物品序列  curseq[物品id1,物品id2,物品id3,.....]
    curseq = sess_clicks[s]
    # 删掉curseq中点击次数小于5的物品id   filseq[物品id1,物品id2,...]  iid_counts{物品id:物品出现次数}
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
    # 如果这个会话中点击次数大于等于5的物品小于两个则删掉
    if len(filseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq      #此时sess_clicks里面存储的一定是物品id大于1个，且每个物品id点击次数不小于5个的字典

print("这是过滤后的sess_clicks")
print(sess_clicks)

# Split out test set based on dates
dates = list(sess_date.items()) #dates[(会话1，年月日),(会话2，年月日)...]
maxdate = dates[0][1]  #年份最大的日子


#再循环确定一下有没有比当前maxdate还要大的年月日
for _, date in dates:
    if maxdate < date:
        maxdate = date

# 7 days for test                   选出一个时间作为训练集和测试集的分界值
splitdate = 0
if opt.dataset == 'yoochoose':
    splitdate = maxdate - 86400 * 1  # the number of seconds for a day：86400
else:
    splitdate = maxdate - 86400 * 7

print('Splitting date', splitdate)      # Yoochoose: ('Split date', 1411930799.0)
tra_sess = filter(lambda x: x[1] < splitdate, dates) #小于tra_sess这个时间的作为训练集 tra_sess[(会话1，年月日),(会话2，年月日)...]
tes_sess = filter(lambda x: x[1] > splitdate, dates) #大于tra_sess这个时间的作为测试集 tes_sess[(会话1，年月日),(会话2，年月日)...]

# Sort sessions by date                没看懂
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
print(len(tra_sess))    # 186670    # 7966257
print(len(tes_sess))    # 15979     # 15324
print("这是训练集")
print(tra_sess)
print("这是测试集")
print(tes_sess)
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
    train_ids = []      # train_ids即训练的sessionid
    train_seqs = []     #train_seqs即这个sessionid点击的物品id train_seqs[[第一个sessionid的物品序列],[第二个sessionid的物品序列],..]
    train_dates = []
    item_ctr = 1
    for s, date in tra_sess:   #s是session号         这个session号为什么是随机的遵循什么规则？
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
    print(item_ctr)     # 43098, 37484               输出值310
    return train_ids, train_dates, train_seqs

# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
    return test_ids, test_dates, test_seqs
tra_ids, tra_dates, tra_seqs = obtian_tra()
tes_ids, tes_dates, tes_seqs = obtian_tes()

#把每一次sessionid点击的最后一个物品当作label标签
#labs是每一个sessionid中点击的最后一个物品
#out_seqs是label前面点击的物品序列
#ids对应out_seqs每一个物品序列在原来seq中的位置
def process_seqs(iseqs, idates):
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [id]
    return out_seqs, out_dates, labs, ids

#tr_seqs是训练集label前面点击的物品序列,与train_seqs(tra_seqs)不同的是tr_seqs会拿出很多重复的序列
#tr_seqs['物品序列1'，'物品序列子1''物品序列2'，'物品序列子2'...'物品序列n']
tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
tra = (tr_seqs, tr_labs)   #前面是会话点击的物品序列，后面是label
tes = (te_seqs, te_labs)   #前面是会话点击的物品序列，后面是label
print(len(tr_seqs))
print(len(te_seqs))
print(tr_seqs[:3], tr_dates[:3], tr_labs[:3])
print(te_seqs[:3], te_dates[:3], te_labs[:3])
all = 0


for seq in tra_seqs:              #tra_seqs['物品序列1'，'物品序列2'...'物品序列n']
    all += len(seq)               #这个all是点击的所有物品个数  len(seq)是n
for seq in tes_seqs:
    all += len(seq)
print('avg length: ', all/(len(tra_seqs) + len(tes_seqs) * 1.0))                 #一个会话的平均长度
if opt.dataset == 'diginetica':
    if not os.path.exists('diginetica'):
        os.makedirs('diginetica')
    pickle.dump(tra, open('diginetica/train.txt', 'wb'))
    pickle.dump(tes, open('diginetica/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('diginetica/all_train_seq.txt', 'wb'))
elif opt.dataset == 'yoochoose':
    if not os.path.exists('yoochoose1_4'):
        os.makedirs('yoochoose1_4')
    if not os.path.exists('yoochoose1_64'):
        os.makedirs('yoochoose1_64')
    pickle.dump(tes, open('yoochoose1_4/test.txt', 'wb'))
    pickle.dump(tes, open('yoochoose1_64/test.txt', 'wb'))

    split4, split64 = int(len(tr_seqs) / 4), int(len(tr_seqs) / 64)
    print(len(tr_seqs[-split4:]))
    print(len(tr_seqs[-split64:]))

    tra4, tra64 = (tr_seqs[-split4:], tr_labs[-split4:]), (tr_seqs[-split64:], tr_labs[-split64:])
    seq4, seq64 = tra_seqs[tr_ids[-split4]:], tra_seqs[tr_ids[-split64]:]

    pickle.dump(tra4, open('yoochoose1_4/train.txt', 'wb'))
    pickle.dump(seq4, open('yoochoose1_4/all_train_seq.txt', 'wb'))

    pickle.dump(tra64, open('yoochoose1_64/train.txt', 'wb'))
    pickle.dump(seq64, open('yoochoose1_64/all_train_seq.txt', 'wb'))

else:
    if not os.path.exists('sample'):
        os.makedirs('sample')
    pickle.dump(tra, open('sample/train.txt', 'wb'))
    pickle.dump(tes, open('sample/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('sample/all_train_seq.txt', 'wb'))

print('Done.')

print("all_train_seq",tra_seqs)
print("train",tra)
print("test",tes)