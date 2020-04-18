import os
import numpy as np
import pandas as pd
from interact.nn_tree import NearestNeighborsTree
from interact import nn_tree
from interact import get_network_from_embedding_using_interact
import time
from interact.nn_tree import NeighborsMode
import textacy
import pickle

## Define important paths
cows_path = '/home/vsharma/Desktop/Clean/COWs_model/'
fast_text_run_path = cows_path + 'fastText-0.2.0/fasttext'
allowed_tree_types_list = ['CORUM_GO','MF_GO','BP_GO','CC_GO','CC_GO_euc','BP_GO_euc','MF_GO_euc',
                           'Disease','Pathway','PC'
                          ]
## Define the name of the tree variables and therefore there pickled file names
tree_var_names = ['pc_tree','go_tree','go_corum_tree',
                  'go_mf_tree','go_bp_tree','go_cc_tree',
                  'go_mf_tree_euc','go_bp_tree_euc','go_cc_tree_euc',
                  'fname_dis_tree','abbr_dis_tree','OMIM_dis_tree','pathways_tree'
                 ]

preprocess = lambda x: textacy.preprocess.preprocess_text(x, lowercase=True, fix_unicode=True,no_punct=True)
def go_name_and_bool_encode(df,go_dag):
    output_dict = {}
    for i,go_id in enumerate(df['GO ID']):
        if go_id in go_dag.keys():
            if go_dag[go_id].namespace == 'biological_process':
                output_dict[i] = {'complex_name':df['complex_name'].iloc[i],'GO ID':go_id,
                                  'GO':preprocess(go_dag[go_id].name),'BP':True,'CC':False,'MF':False}
            elif go_dag[go_id].namespace == 'cellular_component':
                output_dict[i] = {'complex_name':df['complex_name'].iloc[i],'GO ID':go_id,
                                  'GO':preprocess(go_dag[go_id].name),'BP':False,'CC':True,'MF':False}
            elif go_dag[go_id].namespace == 'molecular_function':
                output_dict[i] = {'complex_name':df['complex_name'].iloc[i],'GO ID':go_id,
                                  'GO':preprocess(go_dag[go_id].name),'BP':False,'CC':False,'MF':True}
            else:
                raise ValueError('{} {} has an incorrect namespace: {}'.format(go_id,
                                                                               go_dag[go_id].name,
                                                                               go_dag[go_id].namespace
                                                                              ))
            
    return pd.DataFrame(output_dict).T.reset_index(drop=True)
class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

def readin_multiquery_file(query_list_filepath,zip_together = True):
    query_list = []
    trees_oi = []
    tree_types = allowed_tree_types_list
    with open(query_list_filepath,'r') as qf:
        for cnt,line in enumerate(qf):
            line = line.replace('\n','')
            split_line = line.split('\t')
            query_list[len(query_list):],trees_oi[len(trees_oi):] = [[x] for x in split_line]
        trees_oi = [item.split(';') for item in trees_oi]
        if not all([set(x).issubset(tuple(tree_types)) for x in trees_oi]):
            lines_idxs_wrong_trees = [i for i, x in enumerate([set(x).issubset(tuple(tree_types)) for x in trees_oi]) if x == False]
            raise ValueError('tree_type must be string item in %s\nLines %i: (%s)'%(tree_types,lines_idxs_wrong_trees,
                                                                                 query_list[lines_idxs_wrong_trees]))
            return
    if not zip_together:
        return(query_list,trees_oi)
    return(list(zip(query_list,trees_oi)))

def get_trees(filepath,tree_var_names, use_cached = True):
    trees_dict = {}
    for tree_name in tree_var_names:
        filename = os.path.join(filepath, tree_name + '.pickle')
        if os.path.isfile(filename)  and use_cached:
            with open(filename,'rb') as f:
                trees_dict[tree_name] = pickle.load(f)
        else:
            trees_dict[tree_name] = make_trees(tree_name)
            with open(filename,'wb') as f:
                pickle.dump(trees_dict[tree_name],f)
    return(trees_dict)
        
def make_trees(tree_name):
    a = {'go_tree' : go_sent_vecs,
         'go_corum_tree' : go_corum_vecs,
         'pc_tree' : prot_complexes_vecs,
         'matteos_tree': matteos
        }[tree_name]
    return nn_tree.NearestNeighborsTree(a)

def create_multiquery_file(query_list,trees_oi_list,filepath_name):
    if not all(isinstance(item, list) for item in [query_list,trees_oi_list]):
        raise ValueError('query_list must be of type list. Please input a list, even if just one query.')
        return
    tree_types = allowed_tree_types_list
    if not all([set(x).issubset(tuple(tree_types)) for x in trees_oi_list]):
        trees_check = [set(x).issubset(tuple(tree_types)) for x in trees_oi_list]
        lines_idxs_wrong_trees = [i for i, x in enumerate(trees_check) if x == False]
        raise ValueError('tree_type must be string item in %s\nProblematic Lines indexes are: %s'%(tree_types,lines_idxs_wrong_trees))
        return

    ## Preprocess text in order to make everything lowercase, fix unicode, and remove punctuation
    query_list = [textacy.preprocess.preprocess_text(x,lowercase=True, fix_unicode=True,no_punct=True)
                  for x in list(query_list)]
    trees_oi_list = [';'.join(i) for i in trees_oi_list]
    with open(filepath_name,'w') as qf:
        for cnt,query in enumerate(query_list):
            qf.write(str(query+'\t'+ trees_oi_list[cnt]+'\n'))

def query_to_vecs(queries,path_to_fasttext,path_to_embedding_bin):
    import pandas as pd
    import textacy
    import os
    import numpy as np
    cwd_path = os.getcwd()
    if not isinstance(queries,list):
        raise ValueError('queries must be inputted as a list. Even if just one query.')
        return
    ## Pre-process queries
    queries = [textacy.preprocess.preprocess_text(x,lowercase=True, fix_unicode=True,no_punct=True) for x in queries]
    queries_tmp_path = os.path.join(cwd_path, 'tmp_queries.txt')
    ## Write out tmp .txt file with query phrases
    with open(queries_tmp_path, 'w') as f:
        for i,item in enumerate(queries):
            #print(i,item)
            f.write(item + '\n')
            ## remember that file2 is the cows_path + 'data/prot_complexes_phrases.txt'
    ## run fasttext to get sentence vectors for each query from cows_model.bin
    queries_vec_out = os.path.join(cwd_path, 'tmp_queries_vec.txt')
    print('%s print-sentence-vectors %s < %s > %s'%(path_to_fasttext,path_to_embedding_bin,queries_tmp_path,queries_vec_out))
    ##################### MAKE SURE TO INCORPORATE ERROR CATCHING AND WARNING THAT YOU MIGHT NEED MORE MEMORY IF GET std::bad_alloc
    os.system('%s print-sentence-vectors %s < %s > %s'%(path_to_fasttext,path_to_embedding_bin,queries_tmp_path,queries_vec_out))
    ## Normalize query sentence vectors
    queries_vecs = pd.read_csv(queries_vec_out,sep = ' ',header = None)
    queries_vecs = queries_vecs.iloc[:,:500]
    queries_vecs.index = list(queries)
    vec_norm = np.sqrt(np.square(np.array(queries_vecs)).sum(axis=1))
    queries_vec_normalized = pd.DataFrame(np.array(queries_vecs) / vec_norm.reshape(len(queries),1),index= queries)
    ## remove tmp file created
    os.system('rm tmp*')
    #queries_vec_normalized.drop_duplicates(inplace = True)
    return(queries_vec_normalized)

def query_to_vecs_gensim(query_phrases,model,meth = 1):
    import numpy as np
    import pandas as pd
    import gensim
    
    l2_norm = lambda df: pd.DataFrame(
        np.array(df) / np.sqrt(np.square(np.array(df)).sum(axis=1)).reshape(
            df.shape[0],1),index = df.index
    )
    idx_query_dict = dict(zip(range(len(query_phrases)),query_phrases))
    idx_query_dict_split = {i: idx_query_dict[i].split() for i in idx_query_dict}
    
    df = pd.DataFrame({(outerKey,val): model.wv[val] for outerKey, split_query in idx_query_dict_split.items()
                       for val in split_query
                      }).T
    df.index.names = ['index','word']
    if meth == 1:
        df_normed = l2_norm(df)
        df_phrases = l2_norm(df_normed.groupby(level=['index']).mean().dropna())
        df_phrases.rename(index=idx_query_dict,inplace=True)
        df_phrases.index.names = [None]
    elif meth == 2:
        df_phrases = l2_norm(df.groupby(level=['index']).mean().dropna())
        df_phrases.rename(index=idx_query_dict,inplace=True)
        df_phrases.index.names = [None]
#     elif meth == 3:
#         #df_phrases = l2_norm(df)
#         df_phrases = df
    return(df_phrases)

def query_tree_get_mqueries_nns(query_vecs_input,trees_dict,tree_type_list,k_nns = 10,**kwargs):
    import pandas as pd
    from interact.nn_tree import NearestNeighborsTree
    output_dict = {}
    if not isinstance(query_vecs_input,pd.core.frame.DataFrame):
        raise ValueError('query_vecs_input needs to be a Pandas DataFrame. Please make it a dataframe and try again.')
        return
    if not isinstance(trees_dict,dict):
        raise ValueError('trees_dict must be a dictionary with key corresponding to tree name tag (e.g. "MF_GO")'
                         ' and value corresponding to the associated tree (e.g. go_mf_tree)'
                        )
        return
    if not all([isinstance(x,list) for x in (tree_type_list)]):
        raise ValueError("\nYou've inputted the wrong type for tree_type_list.\n"
                         "Input should be list of lists corresponding to trees you want run.\n"
                         "Please consider first using multi query file reader as input before this step."
                        )
        return    
#     allowed_tree_types = {'MF_GO': go_mf_tree,'BP_GO': go_bp_tree,'CC_GO': go_cc_tree,
#                           'MF_GO_euc': go_mf_tree_euc,'BP_GO_euc': go_bp_tree_euc,'CC_GO_euc': go_cc_tree_euc,
#                           'Disease': fname_dis_tree,'Pathway': pathways_tree,'PC': pc_tree,
#                           'CORUM_GO': go_corum_tree
#                          }
    allowed_tree_types = {'MF_GO': 'go_mf_tree','BP_GO': 'go_bp_tree','CC_GO': 'go_cc_tree',
                          'MF_GO_euc': 'go_mf_tree_euc','BP_GO_euc': 'go_bp_tree_euc','CC_GO_euc': 'go_cc_tree_euc',
                          'Disease': 'fname_dis_tree','Pathway': 'pathways_tree','PC': 'pc_tree',
                          'CORUM_GO': 'go_corum_tree'
                         }
    if not all([set(x).issubset(tuple(allowed_tree_types.keys())) for x in tree_type_list]):
        trees_check = [set(x).issubset(tuple(allowed_tree_types.keys())) for x in tree_type_list]
        lines_idxs_wrong_trees = [i for i, x in enumerate(trees_check) if x == False]
        raise ValueError('tree_type must be string item in %s\n'
                         'Problematic Lines indexes are: %s'%(list(allowed_tree_types.keys()),lines_idxs_wrong_trees)
                        )
        return
    if 'go_dag' in kwargs:
        go_dag = kwargs['go_dag']
        go_map_full = pd.DataFrame([
            (preprocess(go_dag[go_id].name),go_id,go_dag[go_id].namespace)
           for go_id in go_dag.keys()],
          columns=['GO','GO ID','class']
        )
        go_map_full_dict = {
            go_map_full['GO'].iloc[i]: go_map_full['GO ID'].iloc[i] for i in range(go_map_full.shape[0])
        }
    
    # retrieve out queries list from input dataframe
    full_queries = list(query_vecs_input.index)
    print(len(full_queries))
    print(len(tree_type_list))
    for tree_type in list(allowed_tree_types.keys()):
        true_idxs = [tree_type in sublist for sublist in tree_type_list]
        tree_type_queries = [item for idx,item in enumerate(full_queries) if true_idxs[idx] == True]
        if len(tree_type_queries) == len(full_queries):
            if k_nns == -1:
                k_nn = len(trees_dict[allowed_tree_types[tree_type]].word_series)
            else:
                k_nn = k_nns
            #print(tree_type,k_nns,k_nn,len(tree_type_queries))
            tmp_nn_list = trees_dict[allowed_tree_types[tree_type]].kneighbors(X=query_vecs_input.loc[tree_type_queries],
                                                                               k=k_nn
                                                                              )
            for i,key in enumerate(list(tree_type_queries)):
                final_df = pd.DataFrame(list(zip(tmp_nn_list[0][i],
                                                 tmp_nn_list[2][i],
                                                 tmp_nn_list[3][i])
                                            ),
                                        columns=['NNs_natlang','NNs_distance','NNs_simil'])
                if 'go_dag' in kwargs:
                    final_df['GO ID'] = final_df['NNs_natlang'].map(go_map_full_dict)
                if key in list(output_dict.keys()):
                    output_dict[key][tree_type] = final_df
                else:
                    output_dict[key] = {tree_type: final_df}
        else:
            continue
    return(output_dict)

def mergedicts(dict1, dict2):
    for k in set(dict1.keys()).union(dict2.keys()):
        if k in dict1 and k in dict2:
            if isinstance(dict1[k], dict) and isinstance(dict2[k], dict):
                yield (k, dict(mergedicts(dict1[k], dict2[k])))
            else:
                # If one of the values is not a dict, you can't continue merging it.
                # Value from second dict overrides one in first and we move on.
                yield (k, dict2[k])
                # Alternatively, replace this with exception raiser to alert you of value conflicts
        elif k in dict1:
            yield (k, dict1[k])
        else:
            yield (k, dict2[k])

def recurse_mergeddicts(*dictn):
    z = {}
    for dict_to_merge in dictn:
        z = dict(mergedicts(z,dict_to_merge))
    return(z)

def get_query_idx(query_string,query_list):
    query_oi_idx = query_list.index(textacy.preprocess.preprocess_text(query_string,
                                                                       lowercase = True,
                                                                      fix_unicode = True,
                                                                      no_punct = True))
    return(query_oi_idx)


def write_cowsresults_tsv(cows_dict,filename,filepath):
    if not all(isinstance(item, str) for item in [filename,filepath]):
        raise ValueError('You must input string for filename and filepath arguments.')
        return
    import pandas as pd
    # To prevent strings from getting truncated when written
    # If finding that text is cutoff, then simply increase the display.max_colwidth parameter
    pd.set_option("display.max_colwidth", 100000)
    with open(os.path.join(filepath,filename + '.tsv'),'w') as f:
        for keys0 in cows_dict:
            for keys1 in cows_dict[keys0]:
                f.write(str('>>>>'+keys0 + ':' + keys1 + '\n'))
                f.write('"col_header"'+'\t'.join(list(cows_dict[keys0][keys1].columns))+'\n')
                for i in range(len(cows_dict[keys0][keys1])):
                    f.write('\t'.join(str(e) for e in list(cows_dict[keys0][keys1].loc[i])) + '\n')
                f.write(str('\\\\' + '\n'))
            f.write(str('\\\\'+'end\n'))
            f.flush()



### Function for reading in output_query results from .tsv file back into Python
def read_cowsresults_tsv(filepath):
    import pandas as pd
    output_dict = {}
    list_to_df = []
    with open(filepath,'r') as fp:
        for cnt, line in enumerate(fp):
            if '>>>>' in line:
                line = line.replace('>>>>','').replace('\n','')
                if line.count(':') > 1:
                    raise ValueError('''Line %i has semicolon in its name, and is messing up the read-in separator. Please Change\nLine %i- "%s"'''%(cnt,cnt,line)
                                    )
                else:
                    split_line = list(line.split(':'))
                    if len(split_line) != 2:
                        raise ValueError('Line %i: length of split_line is not 2: length %i\n"%s"'%(cnt,len(split_line),line))
                        return
                    else:
                        key0_tmp,key1_tmp = split_line
                        keys0 = str(key0_tmp)
                        keys1 = str(key1_tmp)
            elif '"col_header"' in line:
                col_header = list(line.replace('"col_header"','').replace('\n','').split('\t'))
            elif ('\\\\' not in line):
                list_to_df.append(line.replace('\n','').split('\t'))
            elif ('\\\\end' not in line):
                if keys0 not in list(output_dict.keys()):
                    output_dict[keys0] = {keys1: pd.DataFrame(list_to_df,columns=col_header)}
                else:
                    output_dict[keys0][keys1] = pd.DataFrame(list_to_df,columns=col_header)
                list_to_df = []
            else:
                continue
    return(output_dict)
            