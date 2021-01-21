import os
import pandas as pd
import numpy as np
import re
import json
from tqdm import tqdm
from scipy import sparse
import gensim.models

class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __init__(self, sentences):
        self.sentences = sentences

    def __iter__(self):
        for line in self.sentences:
            # assume there's one document per line, tokens separated by whitespace
            yield line.strip().split(' ')


class Node2Vec():
    def __init__(self,label_path, mat_dict, num_sentence, num_tokens, vec_size,output_path,p=0.5,q=0.5):
        self.label_path = label_path
        self.mat_dict = mat_dict
        self.num_sentence = num_sentence
        self.num_tokens = num_tokens
        self.vec_size = vec_size
        self.output_path = output_path
        self.p = p 
        self.q = q

    def perform_walk(self):
        sentences = []
        if 'AA' in mat_dict:
            self.A_train = sparse.load_npz(mat_dict['AA']).tocsr()
            for _ in tqdm(range(self.num_sentences)):  
                sentence_len = np.random.choice(self.num_tokens)
                A_row = self.A_train.shape[0]
                A_col = self.A_train.shape[1]
                sentence = ''
                app = np.random.choice(A_row)
                api = np.random.choice(np.nonzero(A_train[start_app,:])[1])
                sentence += f'app{app} api{api} ' 
                for i in range(self.sentence_len-1):
                    app = AA_generate_with_probability(app,api,'api')
                    api = AA_generate_with_probability(app,api,'app')
                    sentence += f'app{app} api{api} ' 
                end_app = AA_generate_with_probability(app,api,'api')
                sentence += f'app{end_app}' 
                sentences.append(sentence)

        corpus = MyCorpus(sentences)
        model = gensim.models.Word2Vec(sentences=corpus, size=vec_size)
        model.save(f'{output_path}/node2vec_AA.model')
    
    def AA_generate_with_probability(self,orig_app, orig_api,start='app'):
        if start == 'app':
            api_list = np.nonzero(A_train[orig_app,:])[1]
            prob_list = [1/self.p if i == orig_api else 1/self.q for i in api_list]
            return np.random.choice(api_list,1,p=prob_list)
        else:
            app_list = np.nonzero(A_train[:,orig_api])[0]
            prob_list = [1/self.p if i==orig_app else 1/self.q for i in app_list]
            return np.random.choice(app_list,1,p=prob_list)

def build_n2v(label_path, mat_dict, num_sentences, num_tokens, vec_size, output_path,p=0.5,q=0.5):
    node2vec = Node2Vec(label_path, mat_dict, num_sentences, num_tokens, vec_size, output_path,p,q)
    node2vec.perform_walk()