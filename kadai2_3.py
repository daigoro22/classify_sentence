from word_and_categ_dict import WordAndCategDict
import chainer
import chainer.links as L
import chainer.functions as F
import pandas as pd
from chainer import initializers, training, iterators, optimizers
from chainer.training import extensions
from chainer import Variable
#import numpy as np
import cupy as np
import argparse
from contextlib import ExitStack
import sys
import matplotlib.pyplot as plt

class LSTMClassifier(chainer.Chain):
    def __init__(self,n_categ,n_embed,n_pred):
        super(LSTMClassifier,self).__init__()
        with self.init_scope():
            self.lstm = L.NStepLSTM(1,n_embed,n_pred,dropout=0.2)
            self.categ = L.Linear(None,n_categ)

    def __call__(self,sentence):
        h2,_,_ = self.lstm(None,None,sentence)
        h3 = F.relu(h2[0])
        h4 = self.categ(h3)
        return h4

class RNNClassifier(chainer.Chain):
    def __init__(self,n_categ,n_embed,n_pred):
        super(RNNClassifier,self).__init__()
        with self.init_scope():
            self.rnn = L.NStepRNNReLU(1,n_embed,n_pred,dropout=0.2)
            self.categ = L.Linear(None,n_categ)
    
    def __call__(self,sentence):
        h2,_ = self.rnn(None,sentence)
        h3 = self.categ(h2[0])
        return h3

class LinearClassifier(chainer.Chain):
    def __init__(self,n_categ,n_embed,n_pred):
        super(LinearClassifier,self).__init__()
        with self.init_scope():
            self.linear = L.Linear(n_embed,n_pred)
            self.categ = L.Linear(None,n_categ)
    
    def __call__(self,sentence):
        h2 = self.linear(sentence[0])
        h3 = self.categ(h2)
        return h3

def dump_graph(graph_dir='result/',_type='loss'):
    file_names = ['result/log_linear','result/log_rnn','result/log_lstm']
    log_df = [pd.read_json(_file)[['validation/main/{}'.format(_type)]] for _file in file_names]
    plot_df = pd.concat(log_df,axis=1)
    plot_df.columns = ['linear','rnn','lstm']
    plot_df.plot(grid=True)
    plt.xlabel('epoch')
    plt.ylabel(_type)
    plt.savefig('{}{}.png'.format(graph_dir,_type))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--epoch',type=int,default=100)
    parser.add_argument('-m','--model', choices=['rnn', 'lstm', 'linear'],default='lstm')
    parser.add_argument('-g','--gpu',choices=['-1','0','1','2','3'],default='0')
    parser.add_argument('-gr','--graph',type=str,default=None)
    args = parser.parse_args()

    if args.graph:
        dump_graph(args.graph,'loss')
        dump_graph(args.graph,'accuracy')
        sys.exit()

    BATCH_SIZE = 60

    model_args = {
        'n_categ':55,
        'n_embed':100,
        'n_pred':100
    }

    def remove_nan(x):
        if x.shape ==(0,):
            return np.zeros((2,100),dtype=np.float32)
        else:
            return x

    def batch_converter(batch,device):
        sentence = [remove_nan(np.array(b[0],dtype=np.float32)) for b in batch]
        category = np.array([b[1] for b in batch],dtype=np.int32)
        return {'sentence':sentence, 'category':category}
    
    def batch_converter_linear(batch,device):
        sentence = [remove_nan(np.array(b[0],dtype=np.float32)) for b in batch]
        sentence = [s.mean(axis=0) for s in sentence]
        category = np.array([b[1] for b in batch],dtype=np.int32)
        return {'sentence':np.array([sentence],dtype=np.float32),'category':category}

    if args.model == 'lstm':
        predictor = LSTMClassifier(**model_args)
        bc = batch_converter
        model_postfix = 'lstm'
    elif args.model == 'rnn':
        predictor = RNNClassifier(**model_args)
        bc = batch_converter
        model_postfix = 'rnn'
    else:
        predictor = LinearClassifier(**model_args)
        bc = batch_converter_linear
        model_postfix = 'linear'

    model = L.Classifier(
        predictor,
        lossfun=F.sigmoid_cross_entropy,
        accfun=F.binary_accuracy,
        label_key='category')
    model.to_gpu()

    wd = WordAndCategDict()

    train_dataset_path = 'corpus/train.pkl'
    test_dataset_path = 'corpus/test.pkl'

    df_train = pd.read_pickle(train_dataset_path)
    df_test = pd.read_pickle(test_dataset_path)
    
    dataset_train = chainer.datasets.TupleDataset(
        df_train['SENTENCE'].values,
        df_train['CATEG'].values)
    dataset_test = chainer.datasets.TupleDataset(
        df_test['SENTENCE'].values,
        df_test['CATEG'].values)

    iter_train = iterators.SerialIterator(dataset_train,BATCH_SIZE,shuffle=True)
    iter_test = iterators.SerialIterator(dataset_test,BATCH_SIZE,shuffle=False,repeat=False)

    optimizer = optimizers.SGD()
    optimizer.setup(model)

    updater = training.StandardUpdater(
        iter_train,
        optimizer,
        device=args.gpu,
        converter=bc,
        loss_func=model
    )

    trainer = training.Trainer(updater,(args.epoch,'epoch'),out='result')
    trainer.extend(extensions.Evaluator(iter_test, model,device=args.gpu,converter=bc))
    trainer.extend(extensions.LogReport(filename='log_'+model_postfix))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss',
        'main/accuracy','validation/main/loss','validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.PlotReport(
        ['main/loss','validation/main/loss'],
        x_key='epoch',
        file_name='loss_{}.png'.format(model_postfix)))
    trainer.extend(extensions.PlotReport(
        ['main/accuracy','validation/main/accuracy'],
        x_key='epoch',
        file_name='accuracy_{}.png'.format(model_postfix)))
    trainer.extend(extensions.ProgressBar())
    trainer.run()