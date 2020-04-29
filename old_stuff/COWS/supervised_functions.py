import os
import numpy as np
import pandas as pd
import sys
import os
root_path = "/home/vsharma/Desktop/Clean/"
COWS_path = os.path.join(root_path,"COWs_model/Scripts/COWS")
sys.path.append(COWS_path)
import time
import textacy
import pickle
from COWS.COWS_functions import *
from COWS.paired_word_dnn_script import *
import json
import functools
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.tree import export_graphviz
from sklearn.metrics import matthews_corrcoef
from scipy import interp
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels
pd.set_option("display.max_colwidth", 100000)
pd.options.mode.chained_assignment = None  # default='warn'

def sample_gen(n, forbid):
    state = dict()
    track = dict()
    for (i, o) in enumerate(forbid):
        x = track.get(o, o)
        t = state.get(n-i-1, n-i-1)
        state[x] = t
        track[t] = x
        state.pop(n-i-1, None)
        track.pop(o, None)
    del track
    for remaining in range(n-len(forbid), 0, -1):
        i = random.randrange(remaining)
        yield state.get(i, i)
        state[i] = state.get(remaining - 1, remaining - 1)
        state.pop(remaining - 1, None)

def reduce_concat(x, sep=""):
    return functools.reduce(lambda x, y: str(x) + sep + str(y), x)

def paste(*lists, sep=" ", collapse=None):
    result = map(lambda x: reduce_concat(x, sep=sep), zip(*lists))
    if collapse is not None:
        return reduce_concat(result, sep=collapse)
    return list(result)

def recursive_items(dictionary):
    for key, value in dictionary.items():
        if type(value) is dict:
            yield from recursive_items(value)
        else:
            yield (key, value)

def get_neg_query_df(pos_combd_df,go_to_complex_GrTr):
    start = time.time()
    index_names_list = [i.split('>>>>') for i in pos_combd_df.index]
    pcs_set = list(set([preprocess(name[0]) for name in index_names_list]))
    gos_set = list(set([preprocess(name[1]) for name in index_names_list]))

    negatives_list = []
    for complex_name in pcs_set:
        for go_descrip in gos_set:
            if not complex_name in go_to_complex_GrTr[go_descrip]:
                #negatives_list.append([preprocess(complex_name),preprocess(go_descrip)])
                negatives_list.append([complex_name,go_descrip])

    neg_query_df = pd.DataFrame(negatives_list,columns=['complex_name','GO'])
    end = time.time()
    print('time taken to generate negatives dictionary: %.2f min'%((end-start)/60))
    return(neg_query_df)

def create_full_models(path_main,mrmr_cols = None,seed = 123):
    ## can input mrmr_cols_chosen into the kwargs dictionary
    #from sklearn.metrics import precision_recall_fscore_support
    from sklearn.utils.multiclass import unique_labels
    #from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.naive_bayes import BernoulliNB
    #from sklearn import svm
    from sklearn.linear_model import LogisticRegression
    for go_class in ['BP','MF','CC']:
        path_stored = os.path.join(path_main,go_class,'stored_dfs')
        files = os.listdir(os.path.join(path_stored))
        files = [x for x in files if x != 'forbid_nums.pickle']
        runs = list(set([int(x.split('.')[0]) for x in files]))
        storing_consol_scores_dict = {}
        labels = np.array([-1,1])
        prec_rec_dict = {}
        rf = lambda: RandomForestClassifier(n_jobs=1, random_state=seed,n_estimators=100)
        nb_gauss = lambda: GaussianNB()
        nb_bernoulli = lambda: BernoulliNB()
        logistic_reg = lambda: LogisticRegression(random_state = seed, solver='lbfgs',max_iter= 1000)
        #svm_linear = lambda: svm.SVC(kernel='linear',random_state=seed,probability=True)
        #svm_rbf = lambda: svm.SVC(kernel='rbf', random_state=seed, gamma = 0.1, C =1.0, probability = True)
        #svm_poly = lambda: svm.SVC(kernel='poly', random_state = seed, C=1.0, gamma=0.1, probability = True)
        for run_n, file_n in zip(runs,files):
            start_run = time.time()
            with open(os.path.join(path_stored,file_n),'rb') as f:
                full_df_n = pickle.load(f) 
            #print(run_n)
            # Labels are the values we want to predict
            y = full_df_n['label']
            # Saving feature names for later use
            if mrmr_cols == None:
                # Remove the labels from the features
                # axis 1 refers to the columns
                X= full_df_n.drop('label', axis = 1)
                X_list = list(full_df_n.columns)
            else:
                if not isinstance(mrmr_cols,list):
                    raise ValueError('if inputting mrmr_cols, it needs to be a list of ints corresponding to columns chosen')
                    return
                X = full_df_n.iloc[:,mrmr_cols]
                X_list = mrmr_cols
            print('shape of X: ',X.shape)

            clfs_dict = {'rf': rf#,
                         #'nb_gauss': nb_gauss,'nb_bernoulli': nb_bernoulli,
                         #'logistic_reg': logistic_reg#,
                         #'dnn': runDNN_supervised(X)
                        }
            for model_type,clf in clfs_dict.items():
                print(go_class,run_n,model_type)
                fit = clf().fit(X,y)
                ## Save models out to .pickle files
                model_path_save = os.path.join(path_main,go_class,'stored_models',
                                               'full_models',model_type+'_'+str(run_n)+'.pickle'
                                              )
                os.makedirs(os.path.dirname(model_path_save), exist_ok=True)
                with open(model_path_save,'wb') as f:
                    pickle.dump(fit,f)

def model_predterms(go_vecs,pc,pc_vec,model):
    '''
    Inputs:
        - go_vecs: vectors of the GO Terms for the relevant class (pd.DataFrame)
        - pc: name of the protein complex of interest (string)
        - pc_vec: vector for pc of interest (numpy array or pd Series)
        - model: trained model you wish to use to predict (e.g. Random Forest Classifier object)
        - GrTr_dict: dictionary of Protein Complex (key): CORUM GO Terms (list of values)
    '''
    import numpy as np
    import pandas as pd
    pc_go_vecs_test = pd.DataFrame(np.hstack([np.array([pc_vec]*go_vecs.shape[0]),
                                              go_vecs]),
                                   index= pc + '>>>>' + go_vecs.index
                                  )

    results_df = pd.DataFrame(model.predict_proba(pc_go_vecs_test),
                              index = go_vecs.index,
                              columns=['neg','pos']
                             )
    results_df = results_df.sort_values(by='pos',ascending=False)    
    return(results_df)

def functional_enrichment(predterms_dict,#=mf_dict_predterms
                          kdtree_dict,#=consol_dict_paralogs
                          go_tree,#=go_dag
                          map_go_dict,#=go_mf_map
                          iloc_cut_dict = {'BP_GO':11044,'MF_GO':5213,'CC_GO':1896}, # values obtained from average NNs needed to recover 100% of ground truth from KDTree search
                          alpha_val = 0.05
                         ):
    from scipy.stats import hypergeom
    from COWS import COWS_functions
    res_dict = COWS_functions.AutoVivification()
    for go_class in ['BP_GO','MF_GO','CC_GO']:
        for n,pc in enumerate(list(kdtree_dict.keys())):
            print(n,": "+pc)
            print(go_class)
            kdtree_rez = kdtree_dict[pc][go_class]
            #kdtree_rez['GO ID'] = kdtree_rez['NNs_natlang'].map(go_map_dict) ## map_go_dict --> {GO_name: GO_ID}
            assoc = {}
            hash_gos = {i:go for i,go in enumerate(predterms_dict[pc][go_class]['combined']['GO ID'])}
            hash_gos_pos = {go:pos for go,pos in zip(predterms_dict[pc][go_class]['combined']['GO ID'],
                                                     predterms_dict[pc][go_class]['combined']['pos']
                                                    )}
            #hash_gos['dummy'] = 'dummy'
            pool = set()
            for i,go_id in hash_gos.items():
                #print(set(go_tree[go_id].get_all_children()))
                assoc[i] = set(go_tree[go_id].get_all_children())
                assoc[i].add(go_id)
                pool.update(list(assoc[i]))
            assoc['dummy'] = set(kdtree_rez['GO ID']) - pool
            ## cut kdtree results to iloc_cut to get 'sample population'
            pop_names = kdtree_rez.iloc[0:iloc_cut_dict[go_class]]
            pop_new = pop_names['GO ID']

            for ii in assoc:
                isSignif = False
                M = len(set(kdtree_rez['NNs_natlang'])) ## Total number of GO terms in MF, BP, or CC set
                n_hyper = len(set.intersection(assoc[ii] , set(kdtree_rez['GO ID']))) ## number of intersections between children terms of ML GO term of interest and full set of GO Terms
                N = len(set(pop_new)) ## Size of sample (should be equal to iloc_cut_dict[go_class])
                if not N == iloc_cut_dict[go_class]:
                    raise ValueError('N should be equal to iloc_cut. Currently N={} and iloc_cut_dict[{}]={}'
                                     '\nPlease check if you have used correct map_go_df or if drop_duplicates has messed up.'.format(
                        N,go_class,iloc_cut_dict[go_class]
                    ))
                successes = set.intersection(assoc[ii],set(pop_new))
                x = len(successes) ## Number of successes
                pval = hypergeom.sf(x-1, M, n_hyper, N)
                ## Bonferroni correction
                alpha_alt = pval/len(assoc.keys())
                #print(alpha_alt)
                alpha_crit = 1 - (1-alpha_alt)**(len(assoc.keys()))
                if alpha_crit < alpha_val: ##Bonferroni correction for multiple testing
                    alpha_crit_str = str(alpha_crit) + '******'
                    isSignif = True
                else:
                    alpha_crit_str = str(alpha_crit)
                if not ii == 'dummy':
                    print('\t{}: {} {} alpha_crit = {}'.format(
                        ii,hash_gos[ii],go_tree[hash_gos[ii]].name,alpha_crit_str
                    )
                         )
                    print('\t\tM = {}; N = {}; n = {}; x = {}'.format(M,N,n_hyper,x))
                    res_dict[pc][go_class][hash_gos[ii]] = {'go_name':preprocess(go_tree[hash_gos[ii]].name),
                                                            'M':M,'N':N,'n_hyper':n_hyper,'x':x,'pval':pval,
                                                            'alpha_alt':alpha_alt,'alpha_crit':alpha_crit,
                                                            'isSignif':isSignif,
                                                            'successes': successes,
                                                            'mapped': pop_names,
                                                            'pos': hash_gos_pos[hash_gos[ii]]
                                                 }
                else:
                    print('\t{} alpha_crit = {}'.format(ii,alpha_crit_str))
                    print('\t\tM = {}; N = {}; n = {}; x = {}'.format(M,N,n_hyper,x))
                    res_dict[pc][go_class][ii] = {'go_name':ii,
                                                  'M':M,'N':N,'n_hyper':n_hyper,'x':x,'pval':pval,
                                                  'alpha_alt':alpha_alt,'alpha_crit':alpha_crit,
                                                  'isSignif':isSignif,
                                                  'successes': successes,
                                                  'mapped': pop_names,
                                                  'pos': np.nan
                                                 }
    return(res_dict)
def go_graph_topchildren(go_dag,parent_term,recs,mapped_success_top10, nodecolor,
                              edgecolor, dpi,
                              draw_parents=True, draw_children=True):
        """Draw AMIGO style network, lineage containing one query record."""
        import pygraphviz as pgv

        grph = pgv.AGraph(name="GO tree")

        edgeset = set()
        for rec in recs:
            if draw_parents:
                edgeset.update(rec.get_all_parent_edges())
            if draw_children:
                edgeset.update(rec.get_all_child_edges())

        edgeset = [(go_dag.label_wrap(a), go_dag.label_wrap(b))
                   for (a, b) in edgeset]

        # add nodes explicitly via add_node
        # adding nodes implicitly via add_edge misses nodes
        # without at least one edge
        for rec in recs:
            grph.add_node(go_dag.label_wrap(rec.item_id))

        for src, target in edgeset:
            # default layout in graphviz is top->bottom, so we invert
            # the direction and plot using dir="back"
            grph.add_edge(target, src)

        grph.graph_attr.update(dpi="%d" % dpi)
        grph.node_attr.update(shape="box", style="rounded,filled",
                              fillcolor="beige", color=nodecolor)
        grph.edge_attr.update(shape="normal", color=edgecolor,
                              dir="forward")#, label="is_a")
        
        children = go_dag[parent_term].get_all_children()
        
        # recs_oi
        recs_oi = [go_dag[go_term_oi] for go_term_oi in mapped_success_top10['GO ID']]
        recs_oi_dict = {go_dag[go_term_oi]:score for go_term_oi,score in zip(mapped_success_top10['GO ID'],mapped_success_top10['NNs_simil'])}
        
        import matplotlib

        cmap = matplotlib.cm.get_cmap('Blues')

        #rgba = cmap(0.5)
        # highlight the query terms
        val_col_map = {}
        for rec in recs:
            print(rec.name)
            try:
                if rec in recs_oi:
                    if rec.name == go_dag[parent_term].name: 
                        val_col_map[rec.name] = matplotlib.colors.rgb2hex('plum')
                        print('parent term: {}'.format(rec.id,rec.name),val_col_map[rec.name]) 
                        node = grph.get_node(go_dag.label_wrap(rec.item_id))
                        node.attr.update(fillcolor=val_col_map[rec.name])
                        
                    else:
                        print(rec.id,rec.name)
                        #val_map[rec] = np.random.uniform(0,1)
                        #value = val_map.get(rec, recs_oi_dict[rec])
                        value = recs_oi_dict[rec]
                        val_col_map[rec.name] = matplotlib.colors.rgb2hex(cmap(recs_oi_dict[rec]))
                        #print(value)
                        node = grph.get_node(go_dag.label_wrap(rec.item_id))
                        node.attr.update(fillcolor=val_col_map[rec.name])
                elif rec.name == go_dag[parent_term].name:
                    val_col_map[rec.name] = matplotlib.colors.rgb2hex('plum')
                    print('parent term: {}'.format(rec.id,rec.name),val_col_map[rec.name]) 
                    node = grph.get_node(go_dag.label_wrap(rec.item_id))
                    node.attr.update(fillcolor=val_col_map[rec.name])
            except:
                continue
        return grph,val_col_map

def sample_negatives(pos_query_df,neg_query_df,pos_combd_df,n_runs,embed_path,fasttext_path,dir_to_make_in = '.',new = True,model = 1):
    import time
    import os
    import pickle
    import pandas as pd
    import numpy as np
    import gensim
    
    os.makedirs(os.path.join(dir_to_make_in,'stored_dfs'),exist_ok= True)
    start_full = time.time()
    storing_dfs_dict = {}
    neg_idx_pool = list(range(neg_query_df.shape[0]))
    #forbid_nums = {}
    run = 0
    if not new:
        files = os.listdir(os.path.join(dir_to_make_in,'stored_dfs'))
        files = [x for x in files if x != 'forbid_nums.pickle']
        run = list(set([int(x.split('.')[0]) for x in files]))[-1] - 1
        print('starting at run %i, since there appear to be files already made'%(run))
        with open(os.path.join(dir_to_make_in,'stored_dfs/forbid_nums.pickle'),'rb') as f:
            forbid_nums = pickle.load(f)
    else:
        forbid_nums = {}
    while run <= n_runs:
        start = time.time()
        print('run number: %i'%(run))
        #print('length of forbid nums: %i'%(len(forbid_nums)))
        sample_ids = []
        complx_names = list(set(neg_query_df['complex_name']))
        idx = 0
        while idx < len(complx_names):
            sub_sample_ids = list(neg_query_df.loc[neg_query_df['complex_name'] == complx_names[idx]].index)
            sub_sample_ids_chosen = []
            ## intialize forbid_nums dictionary with complex name if it doesn't previously exist in dict
            if not complx_names[idx] in forbid_nums.keys():
                forbid_nums[complx_names[idx]] = []
            for i in sample_gen(len(sub_sample_ids),forbid_nums[complx_names[idx]]):
                if random.uniform(0,1) <= (pos_query_df.shape[0]/neg_query_df.shape[0]):
                    sub_sample_ids_chosen.append(i)
            if not len(sub_sample_ids_chosen) == 0:
                for j in sub_sample_ids_chosen:
                    sample_ids.append(sub_sample_ids[j])
                    forbid_nums[complx_names[idx]].append(j)
                idx += 1
        print('length of negative sample ids: %i'%(len(sample_ids)))
        if model == 1:
            neg_complex_vecs =  query_to_vecs(list(neg_query_df['complex_name'].iloc[sample_ids]),
                                          path_to_fasttext=fasttext_path,
                                          path_to_embedding_bin=embed_path
                                         )

            neg_go_vecs =  query_to_vecs(list(neg_query_df['GO'].iloc[sample_ids]),
                                         path_to_fasttext=fasttext_path,
                                         path_to_embedding_bin=embed_path
                                        )
        else:
            neg_complex_vecs =  query_to_vecs_gensim(list(neg_query_df['complex_name'].iloc[sample_ids]),
                                         model=model,
                                         meth=1
                                         )

            neg_go_vecs =  query_to_vecs_gensim(list(neg_query_df['GO'].iloc[sample_ids]),
                                         model=model,
                                         meth=1
                                        )
        neg_new_idx_labs = paste(list(neg_complex_vecs.index), list(neg_go_vecs.index), sep='>>>>')
        neg_combd_df = pd.DataFrame(np.hstack((np.array(neg_complex_vecs),np.array(neg_go_vecs))),
                                    index=neg_new_idx_labs)
        neg_combd_df['label'] = -1
        full_combd_df = pd.concat([pos_combd_df,neg_combd_df])
        full_combd_df = full_combd_df.sample(frac=1)  ## shuffle rows
        print(full_combd_df.shape)
        full_combd_df.to_pickle(os.path.join(dir_to_make_in,'stored_dfs',str(run)+'.pickle'))
        with open(os.path.join(dir_to_make_in,'stored_dfs','forbid_nums.pickle'),'wb') as f:
            pickle.dump(forbid_nums,f)
        storing_dfs_dict[run] = full_combd_df
        end = time.time()
        print('time taken for this run %.2f min'%((end-start)/60))
        #return(forbid_nums)
        run += 1
    end_full = time.time()
    print('full time for %i runs: %.2f min'%(n_runs,(end_full-start_full)/60))
    return(storing_dfs_dict)

def scoring_coverage(go_vecs,pc,pc_vec,model,GrTr_dict):
    '''
    Inputs:
        - go_vecs: vectors of the GO Terms for the relevant class (pd.DataFrame)
        - pc: name of the protein complex of interest (string)
        - pc_vec: vector for pc of interest (numpy array or pd Series)
        - model: trained model you wish to use to predict (e.g. Random Forest Classifier object)
        - GrTr_dict: dictionary of Protein Complex (key): CORUM GO Terms (list of values)
    '''
    import numpy as np
    import pandas as pd
    pc_go_vecs_test = pd.DataFrame(np.hstack([np.array([pc_vec]*go_vecs.shape[0]),
                                              go_vecs]),
                                   index= pc + '>>>>' + go_vecs.index
                                  )

    results_df = pd.DataFrame(model.predict_proba(pc_go_vecs_test),
                              index = go_vecs.index,
                              columns=['neg','pos']
                             )
    #print(results_df.sort_values(by = 'pos',ascending = False).head())
    results_filt_df = results_df[results_df['pos'] > 0.5].sort_values(by = 'pos',ascending = False)
    GT_items = GrTr_dict[pc]
    len_GT_items = len(GT_items)
    score = (len(set(results_filt_df.index)& set(GT_items))/len(set(GT_items)))*100
    nns_list = list(results_filt_df.index)
    n_neighbors = [j for j in range(len(nns_list)) if nns_list[j] in GT_items]
    if not score == 0:
        tmp_dict = {'nns': [int(num) for num in np.array(n_neighbors) + 1],
                    'first_nn': int(n_neighbors[0]+1),
                    'n_neighbors_needed': int(n_neighbors[-1]+1),
                    'nn_coverage_score': len_GT_items/(n_neighbors[-1]+1),
                    'len_GT_items': len_GT_items,
                    'GT_items': GT_items,
                    'percent_GT_covered': score
                   }
    else:
        tmp_dict = {'nns': np.nan,
                    'first_nn': np.nan,
                    'n_neighbors_needed': np.nan,
                    'nn_coverage_score': np.nan,
                    'len_GT_items': len_GT_items,
                    'GT_items': GT_items,
                    'percent_GT_covered': score
                   }
    print('{} prediction coverage score of GrTr GO Terms = {}'.format(pc,score))
    return(tmp_dict,results_filt_df) ## tmp_dict is dictionary with relevant summary scores

def pc_leave_out_scoring(idx_names_split,go_vecs,GrTr_dict,model_out_path,run_num,X,y,classifiers,seed = 123,save_model = False):
    start = time.time()
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_curve, auc, matthews_corrcoef
    from scipy import interp
    import numpy as np
    import pickle
    import json
    pc_names = [name[0] for idx,name in idx_names_split.items()]
    go_names = [name[1] for idx,name in idx_names_split.items()]

    pc_leave_out = {}
    for pc_out in list(set(pc_names)):    
        bool_list = []
        for pc in pc_names:
            bool_list.append(pc_out == pc)
        pc_leave_out[pc_out] = bool_list
    cv_report_dict = {}
    for keys in classifiers:
        cv_report_dict[keys] = {}
    labels = np.array([-1,1])
    mean_fpr = np.linspace(0,1,100)
    for idx_pc, (pc_out, bool_list) in enumerate(pc_leave_out.items()):
        print(pc_out)
        X_train = X[[not i for i in bool_list]]
        X_test = X[bool_list]
        y_train = y[[not i for i in bool_list]]
        y_test = y[bool_list]
        #print('test_df shape = {}'.format(X_test.shape))
        pc_vec = X_test.iloc[0,:500]
        sys.stdout.flush()
        if 1 in set(y_test) and -1 in set(y_test):
            #print(pc_out)
            #print(1 in set(y_test) and -1 in set(y_test))
            predict_labels,predict_probs,true_labels,acc,tprs,aucs,mcc,precision,recall,support,f1_score = ([]
                                                                                                            for i in
                                                                                                            range(11)
                                                                                                           )
            for cnt_clf, (key,clf) in enumerate(classifiers.items()):
#                 if key == 'rf':
#                     save_model = True
#                 else:
#                     save_model = False
                if key == 'dnn':  ## Have to convert negative label to 0 for DNN
                    y_train[y_train == -1] = 0
                    y_test[y_test == -1] = 0
                    labels = np.array([0,1])
                    clf = lambda: runDNN_supervised(X)
                else:
                    labels = np.array([-1,1])
                print(list(y_test))
                fit = clf().fit(X_train,y_train)
                prediction = fit.predict_proba(X_test)
                #print(list(prediction))
                predict_probs.append(prediction)
                if key == 'dnn':
                    tmp_pred_labs = torch.tensor(fit.prediction > 0.5)
                    print(tmp_pred_labs)
                    predict_labels.append([val[0] for val in tmp_pred_labs.tolist()])
                    acc.append(
                        (
                            tmp_pred_labs.numpy().flatten() == torch.tensor(y_test).numpy().flatten()
                        ).astype(int).mean()
                    )
                else:
                    predict_labels.append(list(fit.predict(X_test)))
                    acc.append(fit.score(X_test,list(y_test)))
                print(fit.score(X_test,list(y_test)))
                true_labels.append(list(y_test))
                if key == 'dnn':
                    fpr, tpr, t = roc_curve(y_test, prediction)
                else:
                    fpr, tpr, t = roc_curve(y_test, prediction[:, 1])
                tprs.append(interp(mean_fpr, fpr, tpr))
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                #print('len of true labels: %i'%(len(true_labels)))
                mcc_val = matthews_corrcoef(true_labels[cnt_clf],predict_labels[cnt_clf])
                mcc.append(mcc_val)
                #print('mcc_val = {}\nmcc[cnt_clf] = {}'.format(mcc_val,mcc[cnt_clf]))
                prec, rec, f1, supp = precision_recall_fscore_support(true_labels[cnt_clf],
                                                                      predict_labels[cnt_clf],
                                                                      labels= labels,
                                                                      average=None)
                ## choosing prec[1] etc since this is the score for the positively labeled vectors
                precision.append(prec[1]);recall.append(rec[1]);f1_score.append(f1[1]);support.append(supp[1])
                
                if key == 'rf':
                    ## Save the models to .pickle files
                    model_path_stuff = os.path.join(model_out_path,'stored_pc_leaveout_results',str(run_num),key)
                    model_path_save = os.path.join(model_path_stuff,'models',str(idx_pc)+'.pickle')
                    pred_terms_name = os.path.join(model_path_stuff,'pred_terms',str(idx_pc)+'.tsv')
                    summ_scores_name = os.path.join(model_path_stuff,'summ_scores',str(idx_pc)+'.json')
                    os.makedirs(os.path.dirname(pred_terms_name), exist_ok=True)
                    os.makedirs(os.path.dirname(summ_scores_name), exist_ok=True)

                    ## Prediction of relevant GO Terms for left out PC trained on rest of data set
                    summ_scores,pred_terms = scoring_coverage(go_vecs=go_vecs,pc=pc_out,
                                                             pc_vec=pc_vec,GrTr_dict=GrTr_dict,
                                                             model=fit
                                                            )
                    ## write out protein complex name with the basename of file 
                    ## (May not be necessary, but just in case for later reading in)
                    with open(os.path.join(model_path_stuff,'filename_pcname_hash.txt'),'a+') as name_hash:
                        name_hash.write(pc_out+'\t'+os.path.basename(summ_scores_name)+'\n')
                    ## write out summary scores for each protein complex to .json file
                    with open(summ_scores_name,'w') as summ_score_file:
                        json.dump(summ_scores,summ_score_file)
                    ## write predicted GO Terms for each protein to .tsv file
                    pred_terms.to_csv(pred_terms_name,sep='\t',header=True,index=True,index_label=pc_out)
                    if save_model:
                        ## If save_model is True, then save trained leave one out models to .pickle file
                        os.makedirs(os.path.dirname(model_path_save), exist_ok=True)
                        with open(model_path_save,'wb') as f:
                            pickle.dump(fit,f)
                
                if key in cv_report_dict.keys():
                    cv_report_dict[key][pc_out] = {
                        'true_labels': true_labels[cnt_clf],
                        'predict_labels': predict_labels[cnt_clf],
                        'mcc': mcc[cnt_clf],
                        'acc': acc[cnt_clf],
                        'aucs': aucs[cnt_clf],
                        'tprs': tprs[cnt_clf],
                        'predict_probs': predict_probs[cnt_clf],
                        'precision_positive': precision[cnt_clf],
                        'recall_positive': recall[cnt_clf],
                        'f1_score_positive': f1_score[cnt_clf],
                        'support_positive': support[cnt_clf],
                        'labels': labels#,
                        #'clf': fit
                    }
                    if 0 in set(y_train):  ## Have to return 0 label to -1
                        y_train[y_train == 0] = -1
                        y_test[y_test == 0] = -1
                else:
                    raise ValueError('Something has gone wrong')
        else:
            continue

    for clf_key in cv_report_dict:
        accs,mccs,aucs,precs,recs,f1s,supps = ([] for i in range(7))
        for pc_out_key in cv_report_dict[clf_key]:
            accs.append(cv_report_dict[clf_key][pc_out_key]['acc'])
            mccs.append(cv_report_dict[clf_key][pc_out_key]['mcc'])
            aucs.append(cv_report_dict[clf_key][pc_out_key]['aucs'])
            precs.append(cv_report_dict[clf_key][pc_out_key]['precision_positive'])
            recs.append(cv_report_dict[clf_key][pc_out_key]['recall_positive'])
            f1s.append(cv_report_dict[clf_key][pc_out_key]['f1_score_positive'])
            supps.append(cv_report_dict[clf_key][pc_out_key]['support_positive'])
        cv_report_dict[clf_key]['means'] = {'mean_acc': np.mean(accs),'mean_auc': np.mean(aucs),
                                            'mean_mcc': np.mean(mccs),'mean_precision_positive': np.mean(precs),
                                            'mean_recall_positive': np.mean(recs),
                                            'mean_f1_score_positive': np.mean(f1s),
                                            'mean_support_positive': np.mean(supps)
                                           }
    end = time.time()
    print('#############################################\nTime taken for this run: %.2f min'%((end-start)/60))
    return(cv_report_dict)

def pc_leave_one_out_cv(stored_df,go_vecs,GrTr_dict,model_out_path,run_num,seed = 123,save_model = True):
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.utils.multiclass import unique_labels
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.naive_bayes import BernoulliNB
    from sklearn import svm
    from sklearn.linear_model import LogisticRegression
    idx_names_split = {idx:name.split('>>>>') for idx,name in enumerate(stored_df.index)}
    X = stored_df.drop('label',axis = 1)
    y = stored_df['label']
    rf = lambda: RandomForestClassifier(n_jobs=1, random_state=seed,n_estimators=100)
    nb_gauss = lambda: GaussianNB()
    nb_bernoulli = lambda: BernoulliNB()
    logistic_reg = lambda: LogisticRegression(random_state = seed, solver='lbfgs',max_iter= 1000)
    #svm_linear = lambda: svm.SVC(kernel='linear',random_state=seed,probability=True)
    #svm_rbf = lambda: svm.SVC(kernel='rbf', random_state=seed, gamma = 0.1, C =1.0, probability = True)
    #svm_poly = lambda: svm.SVC(kernel='poly', random_state = seed, C=1.0, gamma=0.1, probability = True)
    clfs_dict = {'rf': rf,
                 'nb_gauss': nb_gauss,'nb_bernoulli': nb_bernoulli,
                 'logistic_reg': logistic_reg#,
                 #'dnn': runDNN_supervised(X)
                }
    classifiers = clfs_dict
    cv_report_dict = pc_leave_out_scoring(idx_names_split=idx_names_split,
                                          go_vecs=go_vecs,GrTr_dict=GrTr_dict,
                                          model_out_path=model_out_path,run_num=run_num,
                                          X=X,y=y,
                                          classifiers = clfs_dict,seed = seed,
                                          save_model = save_model
                                         )
    return(cv_report_dict)

## Write out scores for each GO Term to tsv file
def write_scores_gos_tsv(scores_dict,dir_path,filename,clf_names = ['rf','nb_gauss','nb_bernoulli','logistic_reg','dnn']):
    import os
    clfs_in_dict = []
    ## Checking to see if clf_names is in the set of classifier keys of dict
    for run_n,clf in scores_dict.items():
        clfs_in_dict.append(clf)
#     if not set(clf_names) <= set(clfs_in_dict):
#         raise ValueError('At least one of the clf_names you have inputted does not exist as classifier key '
#                         'in the scores_dict. So far ["rf","nb_gauss"] are supported.')
#         return
    
    for clf in clf_names:
        os.makedirs(os.path.join(dir_path,clf), exist_ok=True)
        with open(os.path.join(dir_path,clf,filename),'w') as f:
            header = ['Run_N','mean_acc','mean_auc','mean_mcc',
                                             'mean_precision','mean_recall','mean_f1_score','mean_support']
            f.write('\t'.join(header) + '\n')
            for run_n in scores_dict:
                print(clf,run_n,clf)
                full_line = '\t'.join([
                    str(run_n),
                    str(scores_dict[run_n][clf]['means']['mean_acc']),
                    str(scores_dict[run_n][clf]['means']['mean_auc']),
                    str(scores_dict[run_n][clf]['means']['mean_mcc']),
                    str(scores_dict[run_n][clf]['means']['mean_precision_positive']),
                    str(scores_dict[run_n][clf]['means']['mean_recall_positive']),
                    str(scores_dict[run_n][clf]['means']['mean_f1_score_positive']),
                    str(scores_dict[run_n][clf]['means']['mean_support_positive']),
                ])
                f.write(full_line + '\n')
        for run_n in scores_dict:
            header = ['Prot_Complex','acc','aucs','mcc','precision',
                      'recall','f1_score','support']
            run_filename = 'Run_'+str(run_n)+'.tsv'
            if not run_filename in os.listdir(os.path.join(dir_path,clf)):
                with open(os.path.join(dir_path,clf,run_filename),'a+') as run_f:
                    run_f.write('\t'.join(header) + '\n')
                    for pc in scores_dict[run_n][clf]:
                        if not pc == 'means':
                            full_line_run = '\t'.join([
                                pc,
                                str(scores_dict[run_n][clf][pc]['acc']),
                                str(scores_dict[run_n][clf][pc]['aucs']),
                                str(scores_dict[run_n][clf][pc]['mcc']),
                                str(scores_dict[run_n][clf][pc]['precision_positive']),
                                str(scores_dict[run_n][clf][pc]['recall_positive']),
                                str(scores_dict[run_n][clf][pc]['f1_score_positive']),
                                str(scores_dict[run_n][clf][pc]['support_positive'])
                            ])
                            run_f.write(full_line_run + '\n')
            else:
                with open(os.path.join(dir_path,clf,run_filename),'a+') as run_f:
                    for pc in scores_dict[run_n][clf]:
                        if not pc == 'means':
                            full_line_run = '\t'.join([
                                pc,
                                str(scores_dict[run_n][clf][pc]['acc']),
                                str(scores_dict[run_n][clf][pc]['aucs']),
                                str(scores_dict[run_n][clf][pc]['mcc']),
                                str(scores_dict[run_n][clf][pc]['precision_positive']),
                                str(scores_dict[run_n][clf][pc]['recall_positive']),
                                str(scores_dict[run_n][clf][pc]['f1_score_positive']),
                                str(scores_dict[run_n][clf][pc]['support_positive'])
                            ])
                            run_f.write(full_line_run + '\n')
                        

#### 5-fold scoring: Full data CV; 70% training set CV; 70% training-30% testing set 
def scoring_full_trained_func(k_folds_cv,X,y,model_out_path,run_num,dat_list_type,classifiers,seed = 123):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_curve, auc, matthews_corrcoef
    from scipy import interp
    import numpy as np
    cv_report_dict = {}
    for keys in classifiers:
        for cnt in range(k_folds_cv):
            cv_report_dict[keys] = {cnt: None}
    mean_fpr = np.linspace(0,1,100)
    cv = StratifiedKFold(n_splits= k_folds_cv,random_state = seed,shuffle=True)
    for cnt_cv,(train,test) in enumerate(cv.split(X,y)):
        start = time.time()
        predict_labels,predict_probs,true_labels,acc,tprs,aucs,mcc,precision,recall,support,f1_score = ([]
                                                                                                        for i in
                                                                                                        range(11)
                                                                                                        )
        for cnt_clf, (key,clf) in enumerate(classifiers.items()):
            #print(key)
            if key == 'dnn':  ## Have to convert negative label to 0 for DNN
                y[y == -1] = 0
                labels = np.array([0,1])
                clf = lambda: runDNN_supervised(X)
            else:
                labels = np.array([-1,1])
            fit = clf().fit(X.iloc[train],y.iloc[train])
            prediction = fit.predict_proba(X.iloc[test])
            predict_probs.append(prediction)
            if key == 'dnn':
                tmp_pred_labs = fit.prediction > 0.5
                predict_labels.append([val[0] for val in tmp_pred_labs.tolist()])
                acc.append(
                    (
                        tmp_pred_labs.numpy().flatten() == torch.tensor(y.iloc[test]).numpy().flatten()
                    ).astype(int).mean()
                )
            else:
                predict_labels.append(list(fit.predict(X.iloc[test])))
                acc.append(fit.score(X.iloc[test],list(y.iloc[test])))
            true_labels.append(list(y.iloc[test]))
            #print(y[test])
            #print(prediction[:,1])
            if key == 'dnn':
                fpr, tpr, t = roc_curve(y[test], prediction)
            else:
                fpr, tpr, t = roc_curve(y[test], prediction[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            #print('len of true labels: %i'%(len(true_labels)))
            mcc.append(matthews_corrcoef(true_labels[cnt_clf],predict_labels[cnt_clf]))
            prec, rec, f1, supp = precision_recall_fscore_support(true_labels[cnt_clf],
                                                                  predict_labels[cnt_clf],
                                                                  labels= labels,
                                                                  average=None)
            ## choosing prec[1] etc since this is the score for the positively labeled vectors
            precision.append(prec[1]);recall.append(rec[1]);f1_score.append(f1[1]);support.append(supp[1])
            
            ## Save models out to .pickle files
            model_path_save = os.path.join(model_out_path,'stored_models',
                                           str(run_num),dat_list_type,
                                           key,str(cnt_cv)+'_fold.pickle'
                                          )
            os.makedirs(os.path.dirname(model_path_save), exist_ok=True)
            with open(model_path_save,'wb') as f:
                pickle.dump(fit,f)
            if key in cv_report_dict.keys():
                cv_report_dict[key][cnt_cv] = {
                    'true_labels': true_labels[cnt_clf],
                    'predict_labels': predict_labels[cnt_clf],
                    'mcc': mcc[cnt_clf],
                    'acc': acc[cnt_clf],
                    'aucs': aucs[cnt_clf],
                    'tprs': tprs[cnt_clf],
                    'predict_probs': predict_probs[cnt_clf],
                    'precision_positive': precision[cnt_clf],
                    'recall_positive': recall[cnt_clf],
                    'f1_score_positive': f1_score[cnt_clf],
                    'support_positive': support[cnt_clf],
                    'labels': labels#,
                    #'clf': fit
                }
                if 0 in set(y):  ## Have to return 0 label to -1
                    y[y == 0] = -1
                #if key == 'rf' and count == 0:
                    #print(cv_report_dict['rf'][0]['precision'])
            else:
                raise ValueError('Something has gone wrong')
        end = time.time()
        #print('Time for %s cv: %.2f'%(cnt_cv,(end-start)/60))
    for clf_key in cv_report_dict:
        accs,mccs,aucs,precs,recs,f1s,supps = ([] for i in range(7))
        for cv_n_key in cv_report_dict[clf_key]:
            accs.append(cv_report_dict[clf_key][cv_n_key]['acc'])
            mccs.append(cv_report_dict[clf_key][cv_n_key]['mcc'])
            aucs.append(cv_report_dict[clf_key][cv_n_key]['aucs'])
            precs.append(cv_report_dict[clf_key][cv_n_key]['precision_positive'])
            recs.append(cv_report_dict[clf_key][cv_n_key]['recall_positive'])
            f1s.append(cv_report_dict[clf_key][cv_n_key]['f1_score_positive'])
            supps.append(cv_report_dict[clf_key][cv_n_key]['support_positive'])
        cv_report_dict[clf_key]['mean_acc'] = np.mean(accs)
        cv_report_dict[clf_key]['mean_auc'] = np.mean(aucs)
        cv_report_dict[clf_key]['mean_mcc'] = np.mean(mccs)
        cv_report_dict[clf_key]['mean_precision_positive'] = np.mean(precs)
        cv_report_dict[clf_key]['mean_recall_positive'] = np.mean(recs)
        cv_report_dict[clf_key]['mean_f1_score_positive'] = np.mean(f1s)
        cv_report_dict[clf_key]['mean_support_positive'] = np.mean(supps)
    return(cv_report_dict)

def scoring_split_test_func(k_folds_cv,X,y,test_features,test_labels,model_out_path,run_num,dat_list_type,classifiers,seed = 123):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_curve, auc, matthews_corrcoef
    from scipy import interp
    import numpy as np
    fit_dict = {}
    cv = StratifiedKFold(n_splits= k_folds_cv,random_state = seed,shuffle=True)
    start = time.time()
    for cnt_clf, (key,clf) in enumerate(classifiers.items()):
        if key == 'dnn':  ## Have to convert negative label to 0 for DNN
            y[y == -1] = 0
        start_clf = time.time()
        #print(key)
        if not key in fit_dict.keys():
            fit = clf().fit(X,y)
            fit_dict[key] = fit
        else:
            fit = fit_dict[key]().fit(X,y)
            fit_dict[key] = fit
        if 0 in set(y):  ## Have to return 0 label to -1
            y[y == 0] = -1
        end_clf = time.time()
        #print('Time taken for training %s: %.2f min'%(key,(end_clf - start_clf)/60))
    cv_report_dict = {}
    mean_fpr = np.linspace(0,1,100)
    for cnt_clf, (key,clf) in enumerate(classifiers.items()):
        if key == 'dnn':
            test_labels[test_labels == -1] = 0
            labels = np.array([0,1])
        else:
            labels = np.array([-1,1])
        prediction = fit_dict[key].predict_proba(test_features)
        true_labels = test_labels
        if key == 'dnn':
            tmp_pred_labs = fit_dict[key].prediction > 0.5
            predict_labels = [val[0] for val in tmp_pred_labs.tolist()]
            acc = (tmp_pred_labs.numpy().flatten() == torch.tensor(true_labels).numpy().flatten()).astype(int).mean()
            fpr, tpr, t = roc_curve(test_labels, prediction)
        else:
            predict_labels = list(fit_dict[key].predict(test_features))
            acc = fit_dict[key].score(test_features,list(true_labels))
            fpr, tpr, t = roc_curve(test_labels, prediction[:, 1])
        
        ## Save trained models to .pickle files
        model_path_save = os.path.join(model_out_path,'stored_models',str(run_num),dat_list_type,key,'split_test_30.pickle')
        os.makedirs(os.path.dirname(model_path_save), exist_ok=True)
        with open(model_path_save,'wb') as f:
            pickle.dump(fit_dict[key],f)
        
        prec, rec, f1, supp = precision_recall_fscore_support(true_labels,
                                                          predict_labels,
                                                          labels= labels,
                                                          average=None)
        cv_report_dict[key] = {
            'true_labels': true_labels,
            'predict_labels': predict_labels,
            'mcc': matthews_corrcoef(true_labels,predict_labels),
            'acc': acc,
            'aucs': auc(fpr, tpr),
            'tprs': interp(mean_fpr, fpr, tpr),
            'predict_probs': prediction,
            'precision_positive': prec[1],
            'recall_positive': rec[1],
            'f1_score_positive': f1[1],
            'support_positive': supp[1],
            'labels': labels#,
            #'clf': fit_dict[key]
        }
        if 0 in set(test_labels):  ## Have to return 0 label to -1
            test_labels[test_labels == 0] = -1
    end = time.time()
    #print('Time for %s cv: %.2f'%(cnt_cv,(end-start)/60))
    return(cv_report_dict)

def run_classifiers(path_stored,model_out_path,mrmr_cols = None,k_folds_cv = 5,seed = 123):
    ## can input mrmr_cols_chosen into the kwargs dictionary
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.utils.multiclass import unique_labels
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.naive_bayes import BernoulliNB
    from sklearn import svm
    from sklearn.linear_model import LogisticRegression

    files = os.listdir(os.path.join(path_stored))
    files = [x for x in files if x != 'forbid_nums.pickle']
    runs = list(set([int(x.split('.')[0]) for x in files]))
    storing_consol_scores_dict = {}
    dat_list = ['full_dat','split_trained_70','split_test_30']
    for cnt in runs:
        for test_dat in dat_list:
            storing_consol_scores_dict[cnt] = {test_dat: None}
    labels = np.array([-1,1])
    prec_rec_dict = {}
    rf = lambda: RandomForestClassifier(n_jobs=1, random_state=seed,n_estimators=100)
    nb_gauss = lambda: GaussianNB()
    nb_bernoulli = lambda: BernoulliNB()
    logistic_reg = lambda: LogisticRegression(random_state = seed, solver='lbfgs',max_iter= 1000)
    #svm_linear = lambda: svm.SVC(kernel='linear',random_state=seed,probability=True)
    #svm_rbf = lambda: svm.SVC(kernel='rbf', random_state=seed, gamma = 0.1, C =1.0, probability = True)
    #svm_poly = lambda: svm.SVC(kernel='poly', random_state = seed, C=1.0, gamma=0.1, probability = True)
    for run_n, file_n in zip(runs,files):
        start_run = time.time()
        with open(os.path.join(path_stored,file_n),'rb') as f:
            full_df_n = pickle.load(f) 
        print(run_n)
        for test_dat in dat_list:
            if test_dat == 'full_dat':
                # Labels are the values we want to predict
                y = full_df_n['label']
                # Saving feature names for later use
                if mrmr_cols == None:
                    # Remove the labels from the features
                    # axis 1 refers to the columns
                    X= full_df_n.drop('label', axis = 1)
                    X_list = list(full_df_n.columns)
                else:
                    if not isinstance(mrmr_cols,list):
                        raise ValueError('if inputting mrmr_cols, it needs to be a list of ints corresponding to columns chosen')
                        return
                    X = full_df_n.iloc[:,mrmr_cols]
                    X_list = mrmr_cols
                print('shape of X: ',X.shape)

                clfs_dict = {'rf': rf,
                             'nb_gauss': nb_gauss,'nb_bernoulli': nb_bernoulli,
                             'logistic_reg': logistic_reg#,
                             #'dnn': runDNN_supervised(X)
                            }
                
                storing_consol_scores_dict[run_n][test_dat] = scoring_full_trained_func(k_folds_cv,X,y,
                                                                                        model_out_path = model_out_path,
                                                                                        run_num = run_n,
                                                                                        dat_list_type = test_dat,
                                                                                        classifiers = clfs_dict,
                                                                                        seed = seed
                                                                                       )
            elif test_dat == 'split_trained_70':
                # Split the data into training and testing sets
                X, test_features, y, test_labels = train_test_split(full_df_n.drop('label', axis = 1),
                                                                                            full_df_n['label'],
                                                                                            test_size = 0.3,
                                                                                            random_state = seed
                                                                                           )
                if mrmr_cols != None:
                    if not isinstance(mrmr_cols,list):
                        raise ValueError('if inputting mrmr_cols, it needs to be a list of '
                                         'ints corresponding to columns chosen')
                        return
                    X = X.iloc[:,mrmr_cols]
                    X_list = mrmr_cols
                print('shape of X: ',X.shape)
                #rf = RandomForestClassifier(n_jobs=-1, random_state=seed,n_estimators=100)
                #nb_gauss = GaussianNB()
                #nb_bernoulli = BernoulliNB()
                #logistic_reg = LogisticRegression(random_state = seed, solver='lbfgs',max_iter= 1000)
                #svm_linear = svm.SVC(kernel='linear',random_state=seed,probability=True)
                #svm_rbf = svm.SVC(kernel='rbf', random_state=seed, gamma = 0.1, C =1.0, probability = True)
                #svm_poly = svm.SVC(kernel='poly', random_state = seed, C=1.0, gamma=0.1, probability = True)
                clfs_dict = {'rf': rf,
                             'nb_gauss': nb_gauss,'nb_bernoulli': nb_bernoulli,
                             'logistic_reg': logistic_reg#,
                             #'dnn': runDNN_supervised(X)
                            }
                storing_consol_scores_dict[run_n][test_dat] = scoring_full_trained_func(k_folds_cv,X,y,
                                                                                        model_out_path = model_out_path,
                                                                                        run_num = run_n,
                                                                                        dat_list_type = test_dat,
                                                                                        classifiers = clfs_dict,
                                                                                        seed = seed
                                                                                       )
            elif test_dat == 'split_test_30':
                # Split the data into training and testing sets
                X, test_features, y, test_labels = train_test_split(full_df_n.drop('label', axis = 1),
                                                                                            full_df_n['label'],
                                                                                            test_size = 0.3,
                                                                                            random_state = seed
                                                                                           )
                if mrmr_cols != None:
                    if not isinstance(mrmr_cols,list):
                        raise ValueError('if inputting mrmr_cols, it needs to be a list of ints corresponding to columns chosen')
                        return
                    X = X.iloc[:,mrmr_cols]
                    X_list = mrmr_cols
                print('shape of X: ',X.shape)
                #rf = RandomForestClassifier(n_jobs=-1, random_state=seed,n_estimators=100)
                #nb_gauss = GaussianNB()
                #nb_bernoulli = BernoulliNB()
                #logistic_reg = LogisticRegression(random_state = seed, solver='lbfgs',max_iter= 1000)
                #svm_linear = svm.SVC(kernel='linear',random_state=seed,probability=True)
                #svm_rbf = svm.SVC(kernel='rbf', random_state=seed, gamma = 0.1, C =1.0, probability = True)
                #svm_poly = svm.SVC(kernel='poly', random_state = seed, C=1.0, gamma=0.1, probability = True)
                clfs_dict = {'rf': rf,
                             'nb_gauss': nb_gauss,'nb_bernoulli': nb_bernoulli,
                             'logistic_reg': logistic_reg#,
                             #'dnn': runDNN_supervised(X)
                            }
                storing_consol_scores_dict[run_n][test_dat] = scoring_split_test_func(k_folds_cv,X,y,
                                                                                      test_features,
                                                                                      test_labels,
                                                                                      model_out_path = model_out_path,
                                                                                      run_num = run_n,
                                                                                      dat_list_type = test_dat,
                                                                                      classifiers = clfs_dict,
                                                                                      seed = seed
                                                                                     )
            else:
                print('No other features are supported at current moment.')

        end_run = time.time()
        print('Time for Run %i: %.2f min'%(run_n,(end_run - start_run)/60))
    return(storing_consol_scores_dict)

def write_scores_kfold_cv_tsv(scores_dict,dir_path,filename,clf_names = ['rf']):
    import os
    clfs_in_dict = []
    ## Checking to see if clf_names is in the set of classifier keys of dict
    for i,j in scores_dict.items():
        for k,l in j.items():
            for m,n in l.items():
                clfs_in_dict.append(m)
    if not set(clf_names) <= set(clfs_in_dict):
        raise ValueError('At least one of the clf_names you have inputted does not exist as classifier key '
                        'in the scores_dict. So far {} are supported.'.format(clfs_in_dict))
        return
    
    for clf in clf_names:
        os.makedirs(os.path.join(dir_path,clf), exist_ok=True)
        with open(os.path.join(dir_path,clf,filename),'w') as f:
            header_full_trained = ['Run_N','mean_acc','mean_auc','mean_mcc',
                                             'mean_precision','mean_recall','mean_f1_score','mean_support']
            header_test = ['Run_N','acc','auc','mcc',
                                     'precision','recall','f1_score','support']
            header0 = [y for x in ['full_dat','split_trained_70','split_test_30'] for y in len(header_test)*[x]]
            f.write('\t'.join(header0) + '\n')
            f.write('\t'.join(['\t'.join(['\t'.join(header_full_trained)]*2),'\t'.join(header_test)]) + '\n')
            for run_n in scores_dict:
                for dat_type in scores_dict[run_n]:
                    print(clf,run_n,dat_type)
                    if dat_type == 'full_dat':
                        full_line = '\t'.join([
                            str(run_n),
                            str(scores_dict[run_n][dat_type][clf]['mean_acc']),
                            str(scores_dict[run_n][dat_type][clf]['mean_auc']),
                            str(scores_dict[run_n][dat_type][clf]['mean_mcc']),
                            str(scores_dict[run_n][dat_type][clf]['mean_precision_positive']),
                            str(scores_dict[run_n][dat_type][clf]['mean_recall_positive']),
                            str(scores_dict[run_n][dat_type][clf]['mean_f1_score_positive']),
                            str(scores_dict[run_n][dat_type][clf]['mean_support_positive']),
                        ])
                    elif dat_type == 'split_trained_70':
                        split_trained_line = '\t'.join([
                            str(run_n),
                            str(scores_dict[run_n][dat_type][clf]['mean_acc']),
                            str(scores_dict[run_n][dat_type][clf]['mean_auc']),
                            str(scores_dict[run_n][dat_type][clf]['mean_mcc']),
                            str(scores_dict[run_n][dat_type][clf]['mean_precision_positive']),
                            str(scores_dict[run_n][dat_type][clf]['mean_recall_positive']),
                            str(scores_dict[run_n][dat_type][clf]['mean_f1_score_positive']),
                            str(scores_dict[run_n][dat_type][clf]['mean_support_positive']),
                        ])
                    elif dat_type == 'split_test_30':
                        split_test_line = '\t'.join([
                            str(run_n),
                            str(scores_dict[run_n][dat_type][clf]['acc']),
                            str(scores_dict[run_n][dat_type][clf]['aucs']),
                            str(scores_dict[run_n][dat_type][clf]['mcc']),
                            str(scores_dict[run_n][dat_type][clf]['precision_positive']),
                            str(scores_dict[run_n][dat_type][clf]['recall_positive']),
                            str(scores_dict[run_n][dat_type][clf]['f1_score_positive']),
                            str(scores_dict[run_n][dat_type][clf]['support_positive']),
                        ])
                    else:
                        raise ValueError('This "%s" data type not supported yet.'
                                         'Implement yourself if necessary'%(dat_type)
                                        )
                f.write('\t'.join([full_line,split_trained_line,split_test_line]) + '\n')

