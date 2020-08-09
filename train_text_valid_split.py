import pandas as pd
from sklearn.model_selection import train_test_split
from word_and_categ_dict import WordAndCategDict
import numpy as np
import argparse
import fasttext
import nltk
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import *
import re
from nltk import word_tokenize

def to_index_and_save(data_s,data_c,wcdict,data_name,index,func=None):
    data_name += '.pkl'
    func = wcdict.stoi if func == None else func

    data_s_new = data_s.map(func)
    data_c_new = data_c.map(wcdict.categ_list_to_i)

    pd.concat([data_s_new,data_c_new],axis=1).to_pickle(data_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ft','--fasttext',type=str,default='fasttext_model/model')
    args = parser.parse_args()

    index=['SENTENCE','CATEG']
    
    wcdict = WordAndCategDict(categ_path='all_categ.txt')

    ma_reuters = LazyCorpusLoader(
        'ma_reuters', CategorizedPlaintextCorpusReader, '(training|test).*',
        cat_file='cats.txt', encoding='ISO-8859-2')

    # Load MA_Reuters
    documents = ma_reuters.fileids()
    # extracting training and testing data (document ID)
    train_docs_id = list(filter(lambda doc: doc.startswith("train"), documents))
    test_docs_id = list(filter(lambda doc: doc.startswith("test"), documents))

    train_docs = [ma_reuters.raw(doc_id) for doc_id in train_docs_id]
    test_docs = [ma_reuters.raw(doc_id) for doc_id in test_docs_id]

    train_docs_categ = [ma_reuters.categories(doc_id) for doc_id in train_docs_id]
    test_docs_categ = [ma_reuters.categories(doc_id) for doc_id in test_docs_id]
    
    '''
    train_docs_categ = [np.array([wcdict.ctoi(c) for c in categ],dtype=np.int32) for categ in train_docs_categ]
    test_docs_categ = [np.array([wcdict.ctoi(c) for c in categ],dtype=np.int32) for categ in test_docs_categ]
    '''

    train = pd.DataFrame([train_docs,train_docs_categ]).transpose()
    test = pd.DataFrame([test_docs,test_docs_categ]).transpose()

    train.columns = index
    test.columns = index

    def tokenize(text): # returning tokens
        min_length = 3
        words = map(lambda word: word.lower(), word_tokenize(text))

        p = re.compile('[a-zA-Z]+')
        filtered_tokens = list(filter (lambda token: p.match(token) and len(token) >= min_length, words))
        return filtered_tokens

    model = fasttext.load_model(args.fasttext)
    func = lambda x:[model[p] for p in tokenize(x)]

    to_index_and_save(
        train['SENTENCE'],
        train['CATEG'],
        wcdict,
        'corpus/train',
        index,
        func)

    to_index_and_save(
        train['SENTENCE'],
        train['CATEG'],
        wcdict,
        'corpus/test',
        index,
        func)

    to_index_and_save(
        train['SENTENCE'],
        train['CATEG'],
        wcdict,
        'corpus/valid',
        index,
        func)