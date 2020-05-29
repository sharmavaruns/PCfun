import pandas as pd
from interact.nn_tree import NearestNeighborsTree


def readin_multiquery_file(query_list_filepath, zip_together=True):
    query_list = []
    trees_oi = []
    tree_types = allowed_tree_types_list
    with open(query_list_filepath, 'r') as qf:
        for cnt, line in enumerate(qf):
            line = line.replace('\n', '')
            split_line = line.split('\t')
            query_list[len(query_list):], trees_oi[len(trees_oi):] = [[x] for x in split_line]
        trees_oi = [item.split(';') for item in trees_oi]
        if not all([set(x).issubset(tuple(tree_types)) for x in trees_oi]):
            lines_idxs_wrong_trees = [i for i, x in enumerate([set(x).issubset(tuple(tree_types)) for x in trees_oi]) if
                                      x == False]
            raise ValueError(
                'tree_type must be string item in %s\nLines %i: (%s)' % (tree_types, lines_idxs_wrong_trees,
                                                                         query_list[lines_idxs_wrong_trees]))
            return
    if not zip_together:
        return (query_list, trees_oi)
    return (list(zip(query_list, trees_oi)))


def get_trees(filepath, tree_var_names, use_cached=True):
    trees_dict = {}
    for tree_name in tree_var_names:
        filename = os.path.join(filepath, tree_name + '.pickle')
        if os.path.isfile(filename) and use_cached:
            with open(filename, 'rb') as f:
                trees_dict[tree_name] = pickle.load(f)
        else:
            trees_dict[tree_name] = make_trees(tree_name)
            with open(filename, 'wb') as f:
                pickle.dump(trees_dict[tree_name], f)
    return (trees_dict)


def make_trees(tree_name):
    a = {'go_tree': go_sent_vecs,
         'go_corum_tree': go_corum_vecs,
         'pc_tree': prot_complexes_vecs,
         'matteos_tree': matteos
         }[tree_name]
    return NearestNeighborsTree(a)

def query_tree_get_mqueries_nns(query_vecs_input ,trees_dict ,tree_type_list ,k_nns = 10 ,**kwargs):
    output_dict = {}
    if not isinstance(query_vecs_input ,pd.core.frame.DataFrame):
        raise ValueError('query_vecs_input needs to be a Pandas DataFrame. Please make it a dataframe and try again.')
    if not isinstance(trees_dict ,dict):
        raise ValueError('trees_dict must be a dictionary with key corresponding to tree name tag (e.g. "MF_GO")'
                         ' and value corresponding to the associated tree (e.g. go_mf_tree)'
                         )
    if not all([isinstance(x ,list) for x in (tree_type_list)]):
        raise ValueError("\nYou've inputted the wrong type for tree_type_list.\n"
                         "Input should be list of lists corresponding to trees you want run.\n"
                         "Please consider first using multi query file reader as input before this step."
                         )
    allowed_tree_types = {'MF_GO': 'go_mf_tree' ,'BP_GO': 'go_bp_tree' ,'CC_GO': 'go_cc_tree',
                          'MF_GO_euc': 'go_mf_tree_euc' ,'BP_GO_euc': 'go_bp_tree_euc' ,'CC_GO_euc': 'go_cc_tree_euc',
                          'Disease': 'fname_dis_tree' ,'Pathway': 'pathways_tree' ,'PC': 'pc_tree',
                          'CORUM_GO': 'go_corum_tree'
                          }
    if not all([set(x).issubset(tuple(allowed_tree_types.keys())) for x in tree_type_list]):
        trees_check = [set(x).issubset(tuple(allowed_tree_types.keys())) for x in tree_type_list]
        lines_idxs_wrong_trees = [i for i, x in enumerate(trees_check) if x == False]
        raise ValueError('tree_type must be string item in %s\n'
                         'Problematic Lines indexes are: %s ' %(list(allowed_tree_types.keys()) ,lines_idxs_wrong_trees)
                         )
    if 'go_dag' in kwargs:
        go_dag = kwargs['go_dag']
        go_map_full = pd.DataFrame([
            (preprocess(go_dag[go_id].name) ,go_id ,go_dag[go_id].namespace)
            for go_id in go_dag.keys()],
            columns=['GO' ,'GO ID' ,'class']
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
        tree_type_queries = [item for idx ,item in enumerate(full_queries) if true_idxs[idx] == True]
        if len(tree_type_queries) == len(full_queries):
            if k_nns == -1:
                k_nn = len(trees_dict[allowed_tree_types[tree_type]].word_series)
            else:
                k_nn = k_nns
            tmp_nn_list = trees_dict[allowed_tree_types[tree_type]].kneighbors \
                (X=query_vecs_input.loc[tree_type_queries],k=k_nn)
            for i ,key in enumerate(list(tree_type_queries)):
                final_df = pd.DataFrame(list(zip(tmp_nn_list[0][i],
                                                 tmp_nn_list[2][i],
                                                 tmp_nn_list[3][i])
                                             ),
                                        columns=['NNs_natlang' ,'NNs_distance' ,'NNs_simil'])
                if 'go_dag' in kwargs:
                    final_df['GO ID'] = final_df['NNs_natlang'].map(go_map_full_dict)
                if key in list(output_dict.keys()):
                    output_dict[key][tree_type] = final_df
                else:
                    output_dict[key] = {tree_type: final_df}
        else:
            continue
    return(output_dict)