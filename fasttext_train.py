import fasttext
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-fm','--fasttext_model',default='fasttext_model/model')
    parser.add_argument('-c','--corpus',default='corpus.txt')
    parser.add_argument('-em','--embed_model',default='skipgram')
    args = parser.parse_args()

    model = fasttext.train_unsupervised(
        args.corpus,
        model=args.embed_model,)
    model.save_model(args.fasttext_model)