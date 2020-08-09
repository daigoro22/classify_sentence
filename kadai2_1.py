import nltk
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import *

if __name__ == "__main__":
    # Loading the corpus
    ma_reuters = LazyCorpusLoader(
        'ma_reuters', CategorizedPlaintextCorpusReader, '(training|test).*',
        cat_file='cats.txt', encoding='ISO-8859-2')

    # Load MA_Reuters
    documents = ma_reuters.fileids()
    print (str(len(documents)) + " total articles")
    # extracting training and testing data (document ID)
    train_docs_id = list(filter(lambda doc: doc.startswith("train"), documents))
    test_docs_id = list(filter(lambda doc: doc.startswith("test"), documents))
    print (str(len(train_docs_id)) + " training data")
    print (str(len(test_docs_id)) + " testing data")
    # Training and testing data
    train_docs = [ma_reuters.raw(doc_id) for doc_id in train_docs_id]
    test_docs = [ma_reuters.raw(doc_id) for doc_id in test_docs_id]
    
    # print the total number of categories
    categories = ma_reuters.categories()
    num_categories = len(categories)
    print (num_categories, " categories")
    print (categories)

    # raw document example（'coffee category')
    # Documents in a category
    category_docs = ma_reuters.fileids("soybean")
    document_id = category_docs[0] # The first document
    # print the inside document
    print (ma_reuters.raw(document_id))