from word_and_categ_dict import WordAndCategDict
import chainer
import chainer.links as L
import chainer.functions as F
import pandas as pd
from chainer import initializers, training, iterators, optimizers
from chainer.training import extensions
from chainer import Variable
#import numpy as np
import numpy as np
import argparse

class DialectClassifier(chainer.Chain):
    
    def __init__(self,n_categ,n_embed,n_lstm):
        super(DialectClassifier,self).__init__()
        with self.init_scope():
            self.lstm = L.NStepLSTM(1,n_embed,n_lstm,dropout=0.2)
            self.categ = L.Linear(None,n_categ)
    
    def sequence_embed(self,embed, xs):
        x_len = [len(x) for x in xs]
        x_section = np.cumsum(x_len[:-1]).astype(np.int32).tolist()
        ex = embed(F.concat(xs, axis=0))
        exs = F.split_axis(ex, x_section, 0)
        return list(exs)

    def __call__(self,dialect):
        h2,_,_ = self.lstm(None,None,dialect)
        h3 = F.relu(h2[0])
        return self.categ(h3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--epoch',type=int,default=100)
    args = parser.parse_args()

    BATCH_SIZE = 60

    model = L.Classifier(DialectClassifier(
        n_categ=55,
        n_embed=100,
        n_lstm=600,
    ),label_key='category')

    wd = WordAndCategDict()

    train_dataset_path = 'corpus/train.pkl'
    test_dataset_path = 'corpus/test.pkl'

    df_train = pd.read_pickle(train_dataset_path)
    df_test = pd.read_pickle(test_dataset_path)

    # print(df_train.head())
    
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

    def batch_converter(batch,device):
        dialect = [np.array(b[0]) for b in batch]
        category = np.array([b[1] for b in batch],dtype=np.int32)
        return {'dialect':dialect,'category':category}

    updater = training.StandardUpdater(
        iter_train,
        optimizer,
        device=0,
        converter=batch_converter,
        loss_func=F.sigmoid_cross_entropy
    )

    trainer = training.Trainer(updater,(args.epoch,'epoch'),out='result')
    trainer.extend(extensions.Evaluator(iter_test, model,device=0,converter=batch_converter))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy',
                                                    'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.PlotReport(
        ['main/loss','validation/main/loss'],
        x_key='epoch',
        file_name='loss.png'))
    trainer.extend(extensions.PlotReport(
        ['main/accuracy','validation/main/accuracy'],
        x_key='epoch',
        file_name='accuracy.png'))
    trainer.extend(extensions.ProgressBar())
    trainer.run()