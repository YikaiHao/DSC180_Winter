import sys
import json
import pandas as pd

sys.path.insert(0, 'src')
from util import *
from data.make_dataset import clean_df
from features.build_features import build_mat
from features.word2vec import build_w2v
from models.run_model import run_model 
from visualization.eda import generate

def load_params(fp):
    """
    Load params from json file 
    """
    with open(fp) as fh:
        param = json.load(fh)

    return param

def main(targets):
    """
    Runs the main project pipeline logic, given the target 
    targets must contain: 'baseline_df' ...
    """
    eda_config = json.load(open('config/eda-params.json'))
    if 'baseline_df' in targets:
        params = load_params('config/data-params.json')
        clean_df(**params)

    if 'eda' in targets:
        params = load_params('config/eda-params.json')
        generate(**params)

    if 'feature_build' in targets:
        params = load_params('config/feature-params.json')
        build_mat(**params)
    
    if 'run_model' in targets:
        params = load_params('config/test-params.json')
        run_model(**params)

    if 'word2vec' in targets:
        params = load_params('config/word2vec.json')
        build_w2v(**params)

    if 'node22vec' in targets:
        params = load_params('config/node2vec.json')
        build_n2v(**params)

    if 'test' in targets:
        params = load_params('config/test/data-params.json')
        clean_df(**params)
        params = load_params('config/test/feature-params.json')
        build_mat(**params)
        params = load_params('config/test/test-params.json')
        run_model(**params)
   

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)