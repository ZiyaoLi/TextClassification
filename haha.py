# coding:utf-8
import pandas as pd
import jieba.analyse

trainx = pd.read_csv("./train_x.csv")
trainy = pd.read_csv("./train_y.csv")
train = pd.DataFrame({
    'cont': trainx['content'],
    'sent': trainy['sentiment']
})

output_pos = open('pos100', 'w')
output_neg = open('neg100', 'w')
output_all = open('all100', 'w')

keywords_pos = jieba.analyse.extract_tags(''.join(train['cont'][train['sent'] == 1]), topK=100)
keywords_neg = jieba.analyse.extract_tags(''.join(train['cont'][train['sent'] == -1]), topK=100)
keywords_all = jieba.analyse.extract_tags(''.join(train['cont'][train['sent'] == 1]), topK=100)
for item in keywords_pos:
    print >> output_pos, item
for item in keywords_neg:
    print >> output_neg, item
for item in keywords_all:
    print >> output_all, item
