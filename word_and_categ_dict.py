from collections import defaultdict
import numpy as np

class WordAndCategDict():
    
    def __init__(self,categ_path='all_categ.txt'):
        with open(categ_path,'r') as f:
            self.__itoc = list(set([c.strip() for c in f]))
        self.__ctoi = {c:i for i,c in enumerate(self.__itoc)}
    
    def get_all_categ(self):
        return self.__itoc

    def ctoi(self,categ):
        return self.__ctoi[categ]
    
    def categ_list_to_i(self,categ_list):
        return np.array([self.__ctoi[categ] for categ in categ_list],dtype=np.int32)
    
    def itoc(self,index):
        return self.__itoc[index]