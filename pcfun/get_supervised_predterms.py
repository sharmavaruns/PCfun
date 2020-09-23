import time
import numpy as np
import pandas as pd
from multiprocessing import Pool


def model_predterms(go_vecs,pc,pc_vec,model):
    '''
    Inputs:
        - go_vecs: vectors of the GO Terms for the relevant class (pd.DataFrame)
        - pc: name of the protein complex of interest (string)
        - pc_vec: vector for pc of interest (numpy array or pd Series)
        - model: trained model you wish to use to predict (e.g. Random Forest Classifier object)
        - GrTr_dict: dictionary of Protein Complex (key): CORUM GO Terms (list of values)
    '''
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

def create_full_models(path_main,mrmr_cols = None,seed = 123):  ## Maybe shouldn't include
    from sklearn.ensemble import RandomForestClassifier
    for go_class in ['BP','MF','CC']:
        path_stored = os.path.join(path_main,go_class,'stored_dfs')
        files = os.listdir(os.path.join(path_stored))
        files = [x for x in files if x != 'forbid_nums.pickle']
        runs = list(set([int(x.split('.')[0]) for x in files]))
        labels = np.array([-1,1])
        rf = lambda: RandomForestClassifier(n_jobs=1, random_state=seed,n_estimators=100)
        for run_n, file_n in zip(runs,files):
            start_run = time.time()
            with open(os.path.join(path_stored,file_n),'rb') as f:
                full_df_n = pickle.load(f)

            # Labels are the values we want to predict
            y = full_df_n['label']

            # Saving feature names for later use
            X = full_df_n.drop('label', axis = 1)
            X_list = list(full_df_n.columns)

            print('shape of X: ',X.shape)

            clfs_dict = {'rf': rf}
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


def get_model_predterms_runner(queries_rez,mf_vecs,bp_vecs,cc_vecs,queries_vecs,supervised_models,name_type):
    ## Get functional predictions for test protein complexes (due to time of running algorithm)
    start = time.time()
    n = 5
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
    return(queries_rez)

def get_model_predterms_norm(queries_rez,queries_vecs,mf_vecs,bp_vecs,cc_vecs,supervised_models,name_type):
    n = 5
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


def get_model_predterms_mp(queries_rez,queries_vecs,*args):#mf_vecs,bp_vecs,cc_vecs,supervised_models,name_type):
    print(f'queries_vecs.shape = {queries_vecs.shape}')
    chunks = np.array_split(queries_vecs,5)

    pool = Pool(processes=5)

    queries_rez = pool.map(get_model_predterms_mp, queries_rez,chunks,*args)

    return(queries_rez)