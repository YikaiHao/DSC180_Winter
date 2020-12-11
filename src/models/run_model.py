import pandas as pd
import numpy as np
import json 
from scipy import sparse
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score

from src.util import *

class model():
    def __init__(self,X_paths,type_list,y_path,metapaths,output_path):
        self.df = pd.DataFrame()
        self.metapaths = metapaths
        self.svms = [svm.SVC(kernel='precomputed') for mp in metapaths]
        self.output_path = output_path

        self._load_matrix(X_paths,type_list)
        self._load_y(y_path)
        self._construct_kernel(metapaths)
        self._evaluate(self.metapaths,self.kernels,self.svms)
        self._save_output()

        

    def _load_matrix(self,X_paths,type_list):
        # load_matrix and save them with their corresponding type of matrix
        for p,t in zip(X_paths,type_list):
            matrix = sparse.load_npz(p)
            if t == "A_train":
                self.A_tr_mat = matrix
            elif t == "A_test":
                self.A_test_mat = matrix 
            elif t == "B_train":
                self.B_tr_mat = matrix 
            elif t == "P_train":
                self.P_tr_mat = matrix
            else :
                raise NotImplementedError 
    
    def _load_y(self, y_path):
        # open the json file
        y_value = json.load(open(y_path))
        self.y_train = y_value['y_train']
        self.y_test = y_value['y_test']


    def _kernel_func(self, metapath):
        # store the matrix 
        A_tr_mat_trans = self.A_tr_mat.T
        B_tr_mat = self.B_tr_mat
        P_tr_mat = self.P_tr_mat

        # return functions for metapath 
        if metapath == "AA":
            func = lambda X: X.dot(A_tr_mat_trans)
        elif metapath == "ABA":
            func = lambda X: (X.dot(B_tr_mat)).dot(A_tr_mat_trans)
        elif metapath == "APA":
            func = lambda X: (X.dot(P_tr_mat)).dot(A_tr_mat_trans)
        elif metapath == "ABPBA":
            func = lambda X: (((X.dot(B_tr_mat)).dot(P_tr_mat)).dot(B_tr_mat.T)).dot(A_tr_mat_trans)
        elif metapath == "APBPA":
            func = lambda X: (((X.dot(P_tr_mat)).dot(B_tr_mat)).dot(P_tr_mat.T)).dot(A_tr_mat_trans)
        else:
            raise NotImplementedError
        return func 
    
    def _construct_kernel(self,metapaths):
        kernel_funcs = []
        for metapath in metapaths:
            kernel_funcs.append(self._kernel_func(metapath))
        self.kernels = kernel_funcs 
    
        
    def _evaluate(self,metapaths,kernels, svms):
        y_train = self.y_train
        y_test = self.y_test 
        X_train = self.A_tr_mat
        X_test = self.A_test_mat
        for mp, kernel, svm in zip(metapaths, kernels, svms):
            print(mp)
            gram_train = kernel(X_train).toarray()
            svm.fit(gram_train,y_train)
            train_acc = svm.score(gram_train, y_train)
            y_pred_train = svm.predict(gram_train)
            f1_tr = f1_score(y_train, y_pred_train)
            tn_tr, fp_tr, fn_tr, tp_tr = confusion_matrix(y_pred_train, y_pred_train).ravel()
            print('Train Accuracy',train_acc)
            print('Train F1', f1_tr)
            print('Train Precision',precision_score(y_train,y_pred_train))
            print('Train Recall',recall_score(y_train,y_pred_train))
            print('Train Data Size', tn_tr+fp_tr+fn_tr+tp_tr)



            gram_test = kernel(X_test).toarray()
            y_pred = svm.predict(gram_test)
            test_acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            print('------')
            print('Test Accuracy',test_acc)
            print('Test F1', f1)
            print('Test Precision',precision_score(y_test,y_pred))
            print('Test Recall',recall_score(y_test,y_pred))
            print('Test Data Size',tn+fp+fn+tp)
            print('---------------------------------')
            self.df[mp] = pd.Series({
                    'train_acc': train_acc, 'test_acc': test_acc,
                    'train_f1':f1_tr,'train_TP':tp_tr,'train_FP':fp_tr,'train_TN':tn_tr,'train_FN':fn_tr,
                    'test_f1': f1,'test_TP': tp, 'test_FP': fp, 'test_TN': tn, 'test_FN': fn
                })
            pd.DataFrame(gram_train).to_csv(f'{self.output_path}/{mp}_train.csv')
            pd.DataFrame(gram_test).to_csv(f'{self.output_path}/{mp}_test.csv')
            
        self.df = self.df.T
        self.df.index = self.df.index.set_names(['metapath'])
    
    def _save_output(self):
        self.df.to_csv(f'{self.output_path}/result.csv')

def run_model(X_paths,type_list,y_path,metapaths,output_path):
    """
    Run model 
    """
    model(X_paths,type_list,y_path,metapaths,output_path)
    print("Result saved")