# coding: utf-8
import pandas as pd
import numpy as np
import sys
import codecs
import jieba
# 实现向量化方法
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# 实现svm和贝叶斯模型
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
# 实现交叉验证
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
# 实现评价指标
from sklearn import metrics
# Doc2Vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import logging

train_x = pd.read_csv("./train_x1.csv")
train_y = pd.read_csv("./train_y1.csv")
test_x = pd.read_csv("./test_x1.csv")

# 设置编码utf-8，并保持stdin，stdout，stderr正常输出。
stdi, stdo, stde = sys.stdin, sys.stdout, sys.stderr
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdin, sys.stdout, sys.stderr = stdi, stdo, stde

class CommentClassifier:
    def __init__(self, classifier_type, vector_type):
        self.classifier_type = classifier_type  # 分类器类型：支持向量机或贝叶斯分类
        self.vector_type = vector_type  # 文本向量化模型：0\1模型,TF模型,TF-IDF模型

    def fit(self, train_x, train_y, max_df, stopwords):
        list_text = list(train_x)

        # 向量化方法：0 - 0/1,1 - TF,2 - TF-IDF
        if self.vector_type == 0:
            self.vectorizer = CountVectorizer(max_df, stop_words=stopwords, ngram_range=(1, 3)).fit(list_text)
        elif self.vector_type == 1:
            self.vectorizer = TfidfVectorizer(max_df, stop_words=stopwords, ngram_range=(1, 3), use_idf=False).fit(
                list_text)
        else:
            self.vectorizer = TfidfVectorizer(max_df, stop_words=stopwords, ngram_range=(1, 3)).fit(list_text)

        self.array_trainx = self.vectorizer.transform(list_text)
        self.array_trainy = train_y

        # 分类模型选择：1 - SVC,2 - LinearSVC,3 - SGDClassifier，三种SVM模型
        if self.classifier_type == 1:
            self.model = SVC(kernel='linear', gamma=10 ** -5, C=1).fit(self.array_trainx, self.array_trainy)
        elif self.classifier_type == 2:
            self.model = LinearSVC().fit(self.array_trainx, self.array_trainy)
        else:
            self.model = SGDClassifier().fit(self.array_trainx, self.array_trainy)

    def predict_value(self, test_x):
        list_text = list(test_x)
        self.array_testx = self.vectorizer.transform(list_text)
        array_predict = self.model.predict(self.array_testx)
        return array_predict

    def predict_proba(self, test_x):
        list_text = list(test_x)
        self.array_testx = self.vectorizer.transform(list_text)
        array_score = self.model.predict_proba(self.array_testx)
        return array_score

# 对数据集的每个样本的文本进行中文分词，如遇到缺失值，使用“还行 一般吧”进行填充

trainx_c = []
for row in train_x.values:
    try:
        raw_words = (" ".join(jieba.cut(row[0])))
        trainx_c.append(raw_words)
    except AttributeError:
        print row[0]
        trainx_c.append(u"还行 一般吧")

trainx_c_array = np.array(trainx_c)

# 生成新数据文件，Comment字段为分词后的内容
train = pd.DataFrame({
    'x': trainx_c_array,
    'y': train_y['category']
})

train_x, test_x, train_y, test_y = train_test_split(train['x'].ravel().astype('U'), train['y'].ravel(),
                                                    test_size=0.2, random_state=4)

stopwords = None

classifier_list = [1, 2, 3]
vector_list = [0, 1, 2]
for classifier_type in classifier_list:
    for vector_type in vector_list:
        commentCls = CommentClassifier(classifier_type, vector_type)
        commentCls.fit(train_x, train_y, 0.98, stopwords)
        if classifier_type == 0:
            value_result = commentCls.predict_value(test_x)
            proba_result = commentCls.predict_proba(test_x)
            print classifier_type, vector_type
            print 'classification report'
            print metrics.classification_report(test_y, value_result, labels=range(0, 14))
            print 'confusion matrix'
            print metrics.confusion_matrix(test_y, value_result, labels=range(0, 14))
        else:
            value_result = commentCls.predict_value(test_x)
            print classifier_type, vector_type
            print 'classification report'
            print metrics.classification_report(test_y, value_result, labels=range(0, 14))
            print 'confusion matrix'
            print metrics.confusion_matrix(test_y, value_result, labels=range(0, 14))


'''
# Doc2Vec Raw Code
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 为train_x列贴上标签"TRAIN"
def labelizeReviews(reviews, label_type):
    labelized = []
    for i, v in enumerate(reviews):
        label = '%s_%s' % (label_type, i)
        labelized.append(TaggedDocument(v.split(" "), [label]))
    return labelized


lab_train_x = labelizeReviews(train_x, "TRAIN")

# 建立Doc2Vec模型model
size = 300
all_data = []
all_data.extend(train_x)

model = Doc2Vec(min_count=1, window=8, size=size, sample=1e-4, negative=5, hs=0, iter=5, workers=8)
model.build_vocab(all_data)

# 设置迭代次数10
for epoch in range(10):
    model.train(train_x)

# 建立空列表pos和neg以对相似度计算结果进行存储，计算每个评论和极好评论之间的余弦距离，并存在pos列表中
# 计算每个评论和极差评论之间的余弦距离，并存在neg列表中
pos = []
neg = []

for i in range(0, len(train_x)):
    pos.append(model.docvecs.similarity("TRAIN_0", "TRAIN_{}".format(i)))
    neg.append(model.docvecs.similarity("TRAIN_1", "TRAIN_{}".format(i)))

# 将pos列表和neg列表更新到原始数据文件中，分别表示为字段PosSim和字段NegSim
data_bi[u'PosSim'] = pos
data_bi[u'NegSim'] = neg

'''