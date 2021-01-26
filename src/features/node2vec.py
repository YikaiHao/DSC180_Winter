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
    def __init__(self,label_path, mat_dict, num_sentence, num_tokens, vec_size,output_path,p,q):
        self.label_path = label_path
        self.mat_dict = mat_dict
        self.num_sentences = num_sentence
        self.num_tokens = num_tokens
        self.vec_size = vec_size
        self.output_path = output_path
        self.p = p 
        self.q = q

    def perform_walk(self):
        sentences = []
        self.A_train = sparse.load_npz(self.mat_dict['A']).tocsr()
        self.B_train = sparse.load_npz(self.mat_dict['B']).tocsr()
        self.P_train = sparse.load_npz(self.mat_dict['P']).tocsr()
        self.H_matrix = (self.B_train+self.P_train).astype(bool).astype(int)
        if 'B' not in self.mat_dict:
            for _ in tqdm(range(self.num_sentences)):  
                sentence_len = np.random.choice(self.num_tokens)
                api_row = self.A_train.shape[0]
                app_col = self.A_train.shape[1]
                sentence = ''
                app = np.random.choice(api_row)
                api = np.random.choice(np.nonzero(self.A_train[app,:])[1])
                sentence += f'app{app} api{api} ' 
                cur = 'api'
                for i in range(sentence_len-1):
                    if cur == 'api':
                        app = self.A_only_generate_with_probability(app,api,'api')
                        cur = 'app'
                        sentence += f'app{app} '
                    else:
                        api = self.A_only_generate_with_probability(app,api,'app')
                        cur = 'api'
                        sentence += f'api{api} '
                end_app = self.A_only_generate_with_probability(app,api,'api')
                sentence += f'app{end_app}' 
                sentences.append(sentence)
        else:
            for _ in tqdm(range(self.num_sentences)):
                sentence_len = np.random.choice(self.num_tokens)
                api_row = self.A_train.shape[0]
                app_col = self.A_train.shape[1]
                sentence = ''
                prev = 'app'+str(np.random.choice(api_row))
                curr = 'api'+str(np.random.choice(np.nonzero(self.A_train[int(prev[3:]),:])[1]))
                sentence += f'{prev} {curr} ' 
                curr_type = 'api'
                prev_type = 'app'
                for i in range(sentence_len-1):
                    curr_type, prev_type,curr,prev = self.generate_with_probability(curr_type, prev_type,curr,prev)
                    sentence += f'{curr} '
                if curr_type == 'api':
                    sentence += 'end'
                else: 
                    while curr_type!='app':
                        curr_type, _place1,_place2,_place3 = self.generate_with_probability(curr_type, prev_type,curr,prev)
                    sentence += f'{curr}'
                sentences.append(sentence)


        corpus = MyCorpus(sentences)
        model = gensim.models.Word2Vec(sentences=corpus, size=self.vec_size)
        model.save(f'{self.output_path}/node2vec_all_100.model')
        print('saved')
    
    def A_only_generate_with_probability(self,orig_app, orig_api,start='app'):
        if start == 'app':
            api_list = np.nonzero(self.A_train[orig_app,:])[1]
            prob_list = [1/self.p if i == orig_api else 1/self.q for i in api_list]
            total_prob = sum(prob_list)
            prob_list = [i/total_prob for i in prob_list]
            return np.random.choice(api_list,p=prob_list)
        else:
            app_list = np.nonzero(self.A_train[:,orig_api])[0]
            prob_list = [1/self.p if i==orig_app else 1/self.q for i in app_list]
            total_prob = sum(prob_list)
            prob_list = [i/total_prob for i in prob_list]
            return np.random.choice(app_list,p=prob_list)

    def generate_with_probability(self,curr_type,prev_type,curr,prev):
            if curr_type == 'app':
                # api-> app -> api 
                curr_list = self.A_train[int(curr[3:]),:].toarray()[0]
                prev_list = self.H_matrix[int(prev[3:]),:].toarray()[0]
                sum_list = np.sum([curr_list, prev_list], axis=0)
                prob_list = []
                potential_list = []
                for ind,val in enumerate(sum_list):
                    potential_list.append('api'+str(ind))
                    if 'api'+str(ind) == prev:
                        prob_list.append(1/self.p)
                    elif 'api'+str(ind) == curr:
                        prob_list.append(0)
                    elif val == 2:
                        prob_list.append(1)
                    elif val == 1: 
                        prob_list.append(1/self.q)
                    else:
                        prob_list.append(0)
                    
                total_prob = sum(prob_list)
                prob_list = [i/total_prob for i in prob_list]
                return_val = np.random.choice(potential_list,p = prob_list)
                return 'api','app',return_val, curr
            else:
                if prev_type == 'app':
                    # app -> api -> api & app-> api -> app
                    # app -> api -> app
                    potential_list = np.nonzero(self.A_train[:,int(curr[3:])])[0]
                    prob_list = [1/self.p if 'app'+str(i)==prev else 1/self.q for i in potential_list]
                    potential_list = ['app'+str(i) for i in potential_list]
                    # app -> api -> api 
                    curr_list = self.H_matrix[int(curr[3:]),:].toarray()[0]
                    prev_list = self.A_train[int(prev[3:]),:].toarray()[0]
                    sum_list = np.sum([curr_list,prev_list], axis=0)
                    for ind, val in enumerate(sum_list):
                        potential_list.append('api'+str(ind))
                        if 'api'+str(ind) == prev:
                            prob_list.append(1/self.p)
                        elif 'api'+str(ind) == curr:
                            prob_list.append(0)
                        elif val == 2:
                            prob_list.append(1)
                        elif val == 1: 
                            prob_list.append(1/self.q)
                        else:
                            prob_list.append(0)
                    total_prob = sum(prob_list)
                    prob_list = [i/total_prob for i in prob_list]
                    return_val = np.random.choice(potential_list,p = prob_list)
                    return return_val[:3],'api',return_val, curr
                else: 
                    # api -> api -> api, api -> api -> app
                    potential_list = []
                    prob_list = []
                    # api -> api -> api 
                    curr_list = self.H_matrix[int(curr[3:]),:].toarray()[0]
                    prev_list = self.H_matrix[int(prev[3:]),:].toarray()[0]
                    sum_list = np.sum([curr_list,prev_list],axis=0)
                    for ind, val in enumerate(sum_list):
                        potential_list.append('api'+str(ind))
                        if 'api'+str(ind) == prev:
                            prob_list.append(1/self.p)
                        elif 'api'+str(ind) == curr:
                            prob_list.append(0)
                        elif val == 2:
                            prob_list.append(1)
                        elif val == 1: 
                            prob_list.append(1/self.q)
                        else:
                            prob_list.append(0)
                    # api -> api -> app
                    curr_list = self.A_train[:,int(curr[3:])].toarray()[0]
                    prev_list = self.A_train[:,int(prev[3:])].toarray()[0]
                    sum_list = np.sum([curr_list, prev_list], axis=0)
                    for ind, val in enumerate(sum_list):
                        potential_list.append('api'+str(ind))
                        if 'app'+str(ind) == prev:
                            prob_list.append(1/self.p)
                        elif 'app'+str(ind) == curr:
                            prob_list.append(0)
                        elif val == 2:
                            prob_list.append(1)
                        elif val == 1: 
                            prob_list.append(1/self.q)
                        else:
                            prob_list.append(0)
                    total_prob = sum(prob_list)
                    prob_list = [i/total_prob for i in prob_list]
                    return_val = np.random.choice(potential_list,p = prob_list)
                    return return_val[:3],'api',return_val, curr

def build_n2v(label_path, mat_dict, num_sentences, num_tokens, vec_size, output_path,p,q):
    node2vec = Node2Vec(label_path, mat_dict, num_sentences, num_tokens, vec_size, output_path,p,q)
    print('start')
    node2vec.perform_walk()
