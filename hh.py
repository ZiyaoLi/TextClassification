class Solution(MLWorker):
    # 请在下面区域作答 #
    def train(self, dataframe_trainx, dataframe_trainy):
        import pandas as pd
        import numpy as np
        import sys
        import codecs
        # 实现向量化方法
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.feature_extraction.text import CountVectorizer
        # 实现svm和贝叶斯模型
        from sklearn.svm import SVC
        from sklearn.svm import LinearSVC
        from sklearn.linear_model import SGDClassifier
        train_x = dataframe_trainx['content']
        train_y = dataframe_trainy['sentiment']
        list_text = list(train_x)
        vectorizer = TfidfVectorizer(max_df=0.98, ngram_range=(1, 3)).fit(list_text)
        array_trainx = vectorizer.transform(list_text)
        array_trainy = train_y
        model = SGDClassifier().fit(array_trainx, array_trainy)
        return model

    def predictValue(self, model, dataframe_testx):
        import pandas as pd
        import numpy as np
        import sys
        import codecs

        # 实现向量化方法
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.feature_extraction.text import CountVectorizer

        # 实现svm和贝叶斯模型
        from sklearn.svm import SVC
        from sklearn.svm import LinearSVC
        from sklearn.linear_model import SGDClassifier

    def predictProbability(self, model, dataframe_testx):
        import pandas as pd
        import numpy as np
        import sys
        import codecs

        # 实现向量化方法
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.feature_extraction.text import CountVectorizer

        # 实现svm和贝叶斯模型
        from sklearn.svm import SVC
        from sklearn.svm import LinearSVC
        from sklearn.linear_model import SGDClassifier
