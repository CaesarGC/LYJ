# LYJ
```
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
import jieba
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

def get_data():
    '''    获取数据，数据的载入    :return: 文本数据，对应的labels    '''
    with open("data/ham_data.txt", encoding="utf8") as ham_f, open("data/spam_data.txt", encoding="utf8") as spam_f:
        ham_data = ham_f.readlines()
        spam_data = spam_f.readlines()
        ham_label = np.ones(len(ham_data)).tolist()
        spam_label = np.zeros(len(spam_data)).tolist()
        corpus = ham_data + spam_data
        labels = ham_label + spam_label
    return corpus, labels
def prepare_datasets(corpus, labels, test_data_proportion=0.3):
    '''    将数据分为训练集和测试集
    :param corpus: 文本数据
    :param labels: label数据
    :param test_data_proportion:测试数据占比
    :return: 训练数据,测试数据，训练label,测试label    '''
    train_X, test_X, train_Y, test_Y = train_test_split(corpus, labels, test_size=test_data_proportion, random_state=42)
    return train_X, test_X, train_Y, test_Y


# 加载停用词
with open("dict/stop_words.utf8", encoding="utf8") as f:
    stopword_list = f.readlines()
def tokenize_text(text):
    tokens = jieba.cut(text)
    tokens = [token.strip() for token in tokens]
    return tokens
def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text
def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ''.join(filtered_tokens)
    return filtered_text
def normalize_corpus(corpus, tokenize=False):
    normalized_corpus = []
    for text in corpus:
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        normalized_corpus.append(text)
        if tokenize:
            text = tokenize_text(text)
            normalized_corpus.append(text)
    return normalized_corpus
def tfidf_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(min_df=1,
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


def get_metrics(true_labels, predicted_labels):
    print(
    'Accuracy:', np.round(metrics.accuracy_score(true_labels,predicted_labels),2))
    print('Precision:', np.round(metrics.precision_score(true_labels,predicted_labels,average='weighted'),2))
    print('Recall:', np.round(metrics.recall_score(true_labels,predicted_labels,average='weighted'),2))
    print('F1 Score:', np.round(metrics.f1_score(true_labels,predicted_labels,average='weighted'),2))

def train_predict_evaluate_model(classifier,train_features, train_labels,test_features, test_labels):
# build model
    classifier.fit(train_features, train_labels)
# predict using model
    predictions = classifier.predict(test_features)
# evaluate model prediction performance
    get_metrics(true_labels=test_labels,predicted_labels=predictions)
    return predictions

def main():
    corpus,labels=get_data()
    corpus,labels=remove_empty_docs(corpus,labels)
    train_corpus,test_corpus,train_labels,test_labels=prepare_datasets(corpus,labels,test_data_proportion=0.3)
    norm_train_corpus=normalize_corpus(train_corpus)
    norm_test_corpus=normalize_corpus(test_corpus)
    tfidf_vectorizer,tfidf_train_features=tfidf_extractor(norm_train_corpus)
    tfidf_test_features=tfidf_vectorizer.transform(norm_test_corpus)
    tokenized_train=[jieba.lcut(text) for text in norm_train_corpus]
    tokenized_test=[jieba.lcut(text) for text in norm_test_corpus]
    svm = SGDClassifier(loss='hinge', n_iter=100)
    print("基于tfidf的支持向量机模型")
    svm_tfidf_predictions = train_predict_evaluate_model(classifier=svm,
                                                     train_features=tfidf_train_features,
                                                     train_labels=train_labels,
                                                     test_features=tfidf_test_features,
                                                     test_labels=test_labels)

```
