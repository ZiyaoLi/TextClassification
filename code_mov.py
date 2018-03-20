# coding: utf-8
import pandas as pd
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics


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
            self.model = SVC(kernel='linear', gamma=10 ** -5, C=5).fit(self.array_trainx, self.array_trainy)
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

trainx = pd.read_csv("./train_x.csv")
trainy = pd.read_csv("./train_y.csv")
train = pd.DataFrame({
    'cont': trainx['content'],
    'sent': trainy['sentiment']
})

train_x, test_x, train_y, test_y = train_test_split(train['cont'].ravel().astype('U'), train['sent'].ravel(),
                                                    test_size=0.2, random_state=4)
with codecs.open('./stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = [item.strip() for item in f]

# stopwords = None

classifier_list = [1, 2, 3]
vector_list = [0, 1, 2]

for classifier_type in classifier_list:
    for vector_type in vector_list:
        commentCls = CommentClassifier(classifier_type, vector_type)
        commentCls.fit(train_x, train_y, 0.98, stopwords)
        value_result = commentCls.predict_value(test_x)
        print classifier_type, vector_type
        print 'classification report'
        print metrics.classification_report(test_y, value_result, labels=[-1, 1])
        print 'confusion matrix'
        print metrics.confusion_matrix(test_y, value_result, labels=[-1, 1])


