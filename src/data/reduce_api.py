import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class tfidf_reduce_api:
    def __init__(self, path, num_apis, output_path):
        self.num_apis = num_apis
        self.output_path = output_path
        self._load_data(path)
        self._run_tfidf_vec(num_apis)
        self._save_reduce_api_csv(output_path, num_apis)
        #self.malware_df
        #self.popular_df
        #self.random_df
        #self.corpus_lst

    def _load_data(self, path):

        # load malware, popular, and random csv 
        self.malware_df = pd.read_csv(f'{path}malware.csv').iloc[:,1:]
        self.popular_df = pd.read_csv(f'{path}popular-apps.csv').iloc[:,1:]
        self.random_df = pd.read_csv(f'{path}random-apps.csv').iloc[:,1:]

        print("Data Loaded")

        # select the useful columns 
        malware = self.malware_df[['api_id', 'app_id']]
        popular = self.popular_df[['api_id', 'app_id']]
        random = self.random_df[['api_id', 'app_id']]

        # get api columns 
        malware_api = malware.groupby('app_id').agg(list)['api_id']
        popular_api = popular.groupby('app_id').agg(list)['api_id']
        random_api = random.groupby('app_id').agg(list)['api_id']

        def generate_corpus(api_series):
            corpus_lst = []
            for api in api_series:
                for i in api:
                    corpus_lst.append(' '.join(['api' + str(j) for j in i]))
            return corpus_lst

        self.corpus_lst = generate_corpus([malware_api, random_api, popular_api])

        print("Corpus Created")

    def _run_tfidf_vec(self, num_apis):
        # initialize vectorizer
        tfIdfVectorizer=TfidfVectorizer(smooth_idf=True)

        # transform the corpus 
        tfIdf = tfIdfVectorizer.fit_transform(self.corpus_lst)

        # get the mean 
        df = pd.DataFrame(np.mean(tfIdf, axis=0).tolist()[0], index=tfIdfVectorizer.get_feature_names(), columns=['tfidf'])

        # sort by tfidf 
        df = df.sort_values('tfidf', ascending=False)

        # save the top #num of apis
        api_lst = df.head(num_apis).index.tolist()

        # remove the 'api' string 
        self.api_lst_num = [int(i[3:]) for i in api_lst]

        print("Top Api List Created")

    def _save_reduce_api_csv(self, output_path, num_apis):
        malware_top = self.malware_df[self.malware_df.api_id.isin(self.api_lst_num )]
        popular_top = self.popular_df[self.popular_df.api_id.isin(self.api_lst_num )]
        random_top = self.random_df[self.random_df.api_id.isin(self.api_lst_num )]

        malware_top.to_csv(f'{output_path}malware_tfidf_{num_apis}.csv')
        popular_top.to_csv(f'{output_path}popular_tfidf_{num_apis}.csv')
        random_top.to_csv(f'{output_path}random_tfidf_{num_apis}.csv')

        print("Result Saved")

def run_reduce_api(path, num_apis, output_path):
    tfidf_reduce_api(path, num_apis, output_path)







