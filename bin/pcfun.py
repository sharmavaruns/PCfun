# !/usr/bin/env python3
"""PCFun executable scipt."""

## dependencies
import os
import time
import pickle
import warnings
import argparse
import pandas as pd
from goatools import obo_parser
import pcfun.mapping as mpng
from pcfun import functional_enrichment
from pcfun import go_dag_functionalities
from interact.nn_tree import NearestNeighborsTree
from pcfun.core import preprocess,AutoVivification
from pcfun.kdtree_nns import query_tree_get_mqueries_nns
from pcfun.get_supervised_predterms import model_predterms



# NOTE: here we should put the main command line script.

def main(embed_path:str,input_dat_path:str,req_inputs_path:str,
         path_obo:str,is_UniProt = False):  ## ,n_clusts = 25): ## because scrapped Functional Enrichment Clust
    start_entire = time.time()
    #use_DCA = True
    embedding_path = embed_path
    sup_models_path = os.path.join(req_inputs_path,'New_FullText')
    abstr_model = mpng.ftxt_model(path_to_fasttext_embedding=embedding_path)
    queries_vecs = abstr_model.tsv_to_vecs(path_to_tsv=input_dat_path,write_vecs=True,vecs_file_prefix='query')
    queries_list = list(queries_vecs.index)

    go_dag = obo_parser.GODag(path_obo)
    go_map = pd.DataFrame([(preprocess(go_dag[go_id].name), go_id, go_dag[go_id].namespace)
                           for go_id in go_dag.keys()],
                          columns=['GO', 'GO ID', 'class'])
    go_map_dict = {go_map['GO'].iloc[i]: go_map['GO ID'].iloc[i] for i in range(go_map.shape[0])}

    ### Make GO vectors
    ## should implement choice to read-in GO vectors from file perhaps, though this doesn't take too long
    mf_terms = pd.DataFrame([(go_id, preprocess(go_dag[go_id].name)) for go_id in go_dag.keys()
                             if go_dag[go_id].namespace == 'molecular_function'], columns=['GO ID', 'GO'])
    mf_vecs = abstr_model.queries_df_to_vecs(list(mf_terms['GO'])).drop_duplicates()

    bp_terms = pd.DataFrame([(go_id, preprocess(go_dag[go_id].name)) for go_id in go_dag.keys()
                             if go_dag[go_id].namespace == 'biological_process'], columns=['GO ID', 'GO'])
    bp_vecs = abstr_model.queries_df_to_vecs(list(bp_terms['GO'])).drop_duplicates()

    cc_terms = pd.DataFrame([(go_id, preprocess(go_dag[go_id].name)) for go_id in go_dag.keys()
                             if go_dag[go_id].namespace == 'cellular_component'], columns=['GO ID', 'GO'])
    cc_vecs = abstr_model.queries_df_to_vecs(list(cc_terms['GO'])).drop_duplicates()

    ############################################################################################################
    ######################### Getting supervised RF results for queries
    ## Loading in UniProt supervised models
    supervised_models = AutoVivification()
    if is_UniProt == True:
        name_type = 'PC_GO_Uniprot'
    else:
        name_type = 'PC_GO_w_complex'
    for go_class in ['BP', 'CC', 'MF']:
        for dat_n in range(5):
            for model_type in ['rf']:
                model_path = os.path.join(sup_models_path, name_type,'full_models',go_class,
                                          model_type + '_' + str(dat_n) + '.pickle'
                                          )
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)

                supervised_models[go_class][dat_n][model_type] = model

    ## Get functional predictions for test protein complexes (due to time of running algorithm)
    start = time.time()
    n = 5
    queries_rez = AutoVivification()
    for i, pc_query in enumerate(list(queries_vecs.index)):  # queries_oi_names[:]):
        for go_class, go_vectors in {'MF': mf_vecs, 'BP': bp_vecs, 'CC': cc_vecs}.items():
            print(i, pc_query, go_class)
            tmp_ls = []
            for run_n in range(n):
                queries_rez[pc_query][go_class+'_GO'][run_n] = model_predterms(go_vectors,
                                                                                        pc_query,
                                                                                        queries_vecs.loc[pc_query],
                                                                                    model=supervised_models[go_class][run_n]['rf']
                                                                                    )
                weight = 1 / n
                tmp_ls.append(queries_rez[pc_query][go_class + '_GO'][run_n]['pos'] * weight)

            ## Get combined score results from the different models trained on each data set
            queries_rez[pc_query][go_class+'_GO']['combined'] = pd.DataFrame(
                sum(tmp_ls).sort_values(ascending=False),
                columns=['pos']
                )
            queries_rez[pc_query][go_class+'_GO']['combined']['GO ID'] = list(
                queries_rez[pc_query][go_class+'_GO']['combined'].index.map(go_map_dict))
            ## Store only values with 'pos' >= 0.5
            queries_rez[pc_query][go_class + '_GO']['combined'] = \
            queries_rez[pc_query][go_class + '_GO']['combined'].loc[
                queries_rez[pc_query][go_class + '_GO']['combined']['pos'] >= 0.5]
    end = time.time()
    print("Time taken for getting supervised RF models' predicted terms: {} min".format(round((end - start) / 60), 3))

    ### Need to include CC QuickGO results into here

    n_query = 0
    print(list(queries_vecs.index)[n_query])
    print(queries_rez[list(queries_vecs.index)[n_query]]['MF_GO']['combined'])
    ############################################################################################################
    ############################################################################################################
    # ### Pre loading manual GO DAG using networkx in case use_DCA is True
    # ## using manual functions for working with GO DAG tree to calculate Wang SemSim as is not in goatools
    # if use_DCA == True:
    #     start = time.time()
    #     warnings.warn(
    #         f'use_DCA == True. I hope you know what you are doing, DCA takes awhile to run, but does'
    #         f' reduce space when plotting tree diagrams for BP_GO.\n'
    #     )
    #     dcas = go_dag_functionalities.run_dca(
    #         path_obo = path_obo,
    #         input_dat_path = input_dat_path,
    #         go_dag = go_dag,
    #         queries_vecs = queries_vecs,
    #         queries_rez = queries_rez,
    #         go_map = go_map,
    #         nclusts=n_clusts
    #     )
    #     queries_rez, queries_rez_original = dcas.runner()
    #
    # end = time.time()
    # print('Time taken for doing Deepest Common Ancestor Clustering: {} min'.format(round((end - start) / 60, 3)))
    ############################################################################################################
    ######################### Getting KDTree nn results for queries
    ## Define the name of the tree variables and therefore there pickled file names
    # tree_var_names = ['pc_tree', 'go_tree', 'go_corum_tree', 'go_mf_tree', 'go_bp_tree', 'go_cc_tree', 'fname_dis_tree',
    #                   'abbr_dis_tree', 'OMIM_dis_tree', 'pathways_tree']
    tree_var_names = ['go_mf_tree', 'go_bp_tree', 'go_cc_tree']
    tree_var_names_dict = dict(zip(tree_var_names,['mf_vecs','bp_vecs','cc_vecs']))
    trees_path = os.path.join(req_inputs_path, 'Trees_NNs')
    for tree_name in tree_var_names:
        if not tree_name+'.pickle' in set(os.listdir(trees_path)):
            start_tree = time.time()
            print(f'Making KDTree for {tree_name}')
            exec(f'{tree_name} = NearestNeighborsTree({tree_var_names_dict[tree_name]})')
            with open(os.path.join(trees_path,tree_name+'.pickle'),'wb') as f:
                pickle.dump(eval(tree_name),f)
            end_tree = time.time()
            print('Time taken %.2f min to build tree for %s' % ((end_tree - start_tree) / 60,
                                                                tree_name + '.pickle'
                                                                ))
    #### Load in kd trees from their pickles
    trees_dict = {}
    for tree_name in tree_var_names:
        with open(os.path.join(trees_path, tree_name + '.pickle'), 'rb') as f:
            tree = pickle.load(f)
            exec('%s = tree' % (tree_name))
            trees_dict[tree_name] = tree
    trees_oi = []
    for i, pc in enumerate(list(queries_vecs.index)):
        # trees_oi.append(['MF_GO', 'BP_GO', 'CC_GO', 'Disease', 'Pathway','PC'])
        trees_oi.append(['MF_GO', 'BP_GO', 'CC_GO'])

    start = time.time()

    consol_dict_test = query_tree_get_mqueries_nns(queries_vecs,
                                                     trees_dict=trees_dict,
                                                     tree_type_list=trees_oi,
                                                     k_nns=-1,
                                                     go_dag=go_dag
                                                     )
    end = time.time()
    print('Time taken to get KDTree nns: %.2f min' % ((end - start) / 60))


    ### Need to write kdtree results to file

    ############################################################################################################

    ############################################################################################################
    ######################### Functional Enrichment analysis
    start = time.time()
    test_funcenrich_rez = functional_enrichment.functional_enrichment(predterms_dict=queries_rez,
                                                kdtree_dict=dict((k, consol_dict_test[k])
                                                                 for k in queries_rez.keys()),
                                                go_tree=go_dag,
                                                map_go_dict=go_map_dict,
                                                iloc_cut_dict={'BP_GO': 11044, 'MF_GO': 5213,
                                                               'CC_GO': 1896},  # CC:1896
                                                alpha_val=0.05
                                                )
    end = time.time()
    print('Time taken for functional enrichment: {} min'.format(round((end - start) / 60, 3)))



    ############################################################################################################

    ############################################################################################################
    ######################### Writing out results
    start = time.time()

    out_rez_path = os.path.join(os.path.dirname(input_dat_path), 'Results')
    os.makedirs(out_rez_path, exist_ok=True)
    for query in list(queries_list):
        for go_class,iloc_cut in {'BP_GO': 11044, 'MF_GO': 5213,'CC_GO': 1896}.items():
            df_out = pd.DataFrame(test_funcenrich_rez[query][go_class]).T.drop(
                ['successes', 'mapped'], axis=1
            ).drop('dummy').sort_values('pval')
            if not df_out.shape[0] == 0:
                print(query,go_class)
                os.makedirs(os.path.join(out_rez_path,query,go_class), exist_ok=True)
                df_out.to_csv(
                    os.path.join(out_rez_path, query,go_class, 'funcenrich_list.tsv'), sep='\t'
                )
                consol_dict_test[query][go_class].iloc[:iloc_cut].to_csv(
                    os.path.join(out_rez_path, query,go_class, 'KDTree_list.tsv'), sep='\t'
                )
            else:
                print(query, go_class,'has no ML predicted terms, only providing KDTree results.')
                os.makedirs(os.path.join(out_rez_path,query,go_class), exist_ok=True)
                consol_dict_test[query][go_class].iloc[:iloc_cut].to_csv(
                    os.path.join(out_rez_path, query,go_class, 'KDTree_list.tsv'), sep='\t'
                )
    end = time.time()
    print('Time taken for writing out results: {} min'.format(round((end - start) / 60, 3)))
    ############################################################################################################
    ############################################################################################################
    ######################### Creating GO Tree Diagrams for functionally enriched terms
    start = time.time()
    tree_diags_plot = ['MF_GO','BP_GO','CC_GO']#'MF_GO','CC_GO'] ##
    counter_cut = 10
    for query in list(queries_vecs.index):
        for go_class,iloc_cut in {'BP_GO': 11044, 'MF_GO': 5213,'CC_GO': 1896}.items():
            if not go_class in tree_diags_plot:
                next
            else:
                counter = 0
                for go_id_parent in test_funcenrich_rez[query][go_class].keys():
                    if not go_id_parent == 'dummy':
                        if test_funcenrich_rez[query][go_class][go_id_parent]['isSignif'] == True:
                            print(query, go_class)
                            os.makedirs(os.path.join(out_rez_path, query, go_class), exist_ok=True)
                            print('{} Tree for:'.format(go_class), query, )
                            name_file = f'{counter+1}___{go_dag[go_id_parent].name}.png'

                            print('\t',go_id_parent, go_dag[go_id_parent].name)
                            succ = test_funcenrich_rez[query][go_class][go_id_parent]['successes']
                            mapped = test_funcenrich_rez[query][go_class][go_id_parent]['mapped']
                            mapped_succ_top10 = mapped.loc[mapped['GO ID'].isin(succ)].iloc[:10]

                            testing_go_recs = {go_dag[go_id]: 'kdtree' for go_id in mapped_succ_top10['GO ID']}
                            testing_go_recs.update({go_dag[go_id_parent]: 'ml_list'})


                            lineage_png_test = os.path.join(out_rez_path, query,go_class,'Tree_diags',name_file)
                            os.makedirs(os.path.dirname(lineage_png_test),exist_ok=True)
                            diagram, diagram_colors = go_dag_functionalities.go_graph_topchildren(
                                go_dag, go_id_parent, testing_go_recs,
                                mapped_success_top10=mapped_succ_top10,
                                nodecolor="blue", edgecolor="lightslateblue", dpi=200, ## Make sure to change as input arg
                                draw_parents=True, draw_children=False
                            )
                            diagram.draw(lineage_png_test, prog='dot')
                            counter += 1
                        if counter > counter_cut:
                            warnings.warn(f'Greater than {counter_cut+1} terms are functionally enriched\n'
                                          f'Only plotting Tree Diagrams for the top {counter_cut+1} most '
                                          f'significant functionally enriched terms.'
                                          )
                            break
    end = time.time()
    print('Time taken for plotting results: {} min'.format(round((end - start) / 60, 3)))

    print('Writing out binary functional enrichment results and full kd tree results')
    with open(os.path.join(out_rez_path,'func_enrich_rez.pickle'),'wb') as f:
        pickle.dump(test_funcenrich_rez,f)
    with open(os.path.join(out_rez_path,'kdtree_rez.pickle'),'wb') as f1:
        pickle.dump(consol_dict_test,f1)

    end_entire = time.time()
    print('Time taken for analysis up to functional enrichment so far to run: {} min'.format(round((end_entire - start_entire) / 60, 3)))

    pass


if __name__ == '__main__':
    """Get k nns from COWS list"""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-e', '--embed_path', type=str, nargs='?', help='path to embedding', required=True)
    parser.add_argument('-i', '--input_dat_path', type=str, help='path to input file with queries', required=True)
    parser.add_argument('-r', '--req_inputs_path', type=str, help='path to directory with required dependency files',
                        required=True)
    parser.add_argument('-o', '--path_obo', type=str, help='path to Gene Ontology .obo file',
                        required=True)
    parser.add_argument("-u",'--is_UniProt', action='store_true', default=False)

    kwargs = vars(parser.parse_args())
    print(kwargs)
    # if kwargs.get('infile') and kwargs.get('query'):
    #     raise ValueError(f"either input a multi query file or single string query")

    # assert not ('infile' in kwargs) & ('query' in kwargs), f"either input a multi query file or single string query"
    main(**kwargs)
    #
    # embedding_path = '/Users/varunsharma/Documents/PCfun_stuff/req_inputs/Embeddings/abstracts_model.bin'
    # input_dat_path = '/Users/varunsharma/Documents/PCfun_stuff/Projects/Test1/input_df.tsv'
    # req_inputs_path = '/Users/varunsharma/Documents/PCfun_stuff/req_inputs'
    # path_obo = '/Users/varunsharma/Documents/PCfun_stuff/req_inputs/go-basic.obo'
    # is_UniProt = False  ## should come from command line flag

    ##### Run following lines in command line from: /Users/varunsharma/PycharmProjects/PCfun
    ##### Assumes is_UniProt flag is not given, thereby setting it to False in python script
    # REL_PWD=/Users/varunsharma/Documents/PCfun_stuff
    # python ./bin/pcfun.py -e $REL_PWD/req_inputs/Embeddings/abstracts_model.bin -i $REL_PWD/Projects/Test1/input_df.tsv -r $REL_PWD/req_inputs -o $REL_PWD/req_inputs/go-basic.obo
