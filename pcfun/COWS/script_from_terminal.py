import os
from COWS_functions import *
import numpy as np
import pandas as pd
import argparse
import time

## Define important paths
cows_path = '/home/vsharma/Desktop/Clean/COWs_model/'
fast_text_run_path = cows_path + 'fastText-0.2.0/fasttext'
prot_complexes_path = cows_path + 'data/Prot_Complex/prot_complexes_phrases_vec_norm.txt'
query_files_path = cows_path + 'data/Query_files/'

os.chdir(cows_path)

def main(infile = '', outfile='', query = '', trees = '', kneighbors = 10) -> None:
    kneighbors = int(kneighbors[0])
    print(infile,kneighbors)
    print(os.getcwd())
    
    ## import subembedding for protein complexes
    prot_complexes_vecs = pd.read_csv(prot_complexes_path,sep = ',',header = None,index_col = 0)
    
    if infile:
        ## read in queries from stored file
        queries,trees_oi = readin_multiquery_file(infile,
                              zip_together = False)
    if query:
        queries = [query]
        trees_oi = [trees.split(';')]
    ## Generate query vectors
    query_pcs_vecs =  query_to_vecs(queries,path_to_fasttext=fast_text_run_path,
                  path_to_embedding_bin=os.path.join(cows_path,'cows_model.bin')
                 )
    
    ## Query KDTrees and get result
    start = time.time()
    consol_dict_test = query_tree_get_mqueries_nns(query_pcs_vecs.loc[queries[:]],tree_type_list = trees_oi,
                                                   k_nns = 10
                                                  )
    end = time.time()
    print('Time taken to create dictionary %.2f min'%((end-start)/60))
    
    start = time.time()
    write_cowsresults_tsv(consol_dict_test,filename = outfile,filepath = query_files_path)
    end = time.time()
    print('Time taken to write output file %.2f min to run'%((end-start)/60))
    
    pass
    
if __name__ == '__main__':
    """Get k nns from COWS list"""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i','--infile', type=str, nargs='?', help='input multi query file')
    parser.add_argument('-o', '--outfile', type=str, help='filename to write output to', required = True)
    parser.add_argument('-q', '--query', type=str, nargs='?', help='if want to make only single query')
    parser.add_argument('-t', '--trees', nargs = '?', help='which trees you want queried for your single query')
    parser.add_argument('-k', '--kneighbors', help = 'how many k nearest neighbors you want queried',nargs = 1,type = str,default = '10',required = True)
    
    kwargs = vars(parser.parse_args())
    print(kwargs)
    if kwargs.get('infile') and kwargs.get('query'):
        raise ValueError(f"either input a multi query file or single string query")
                                           
    #assert not ('infile' in kwargs) & ('query' in kwargs), f"either input a multi query file or single string query"
    main(**kwargs)



