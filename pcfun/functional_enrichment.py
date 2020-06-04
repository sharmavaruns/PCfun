from scipy.stats import hypergeom
from pcfun import core
import numpy as np


def functional_enrichment(predterms_dict,  # =mf_dict_predterms
                          kdtree_dict,  # =consol_dict_paralogs
                          go_tree,  # =go_dag
                          map_go_dict,  # =go_mf_map
                          iloc_cut_dict={'BP_GO': 11044, 'MF_GO': 5213, 'CC_GO': 1896},
                          # values obtained from average NNs needed to recover 100% of ground truth from KDTree search
                          alpha_val=0.05
                          ):
    res_dict = core.AutoVivification()
    for go_class in ['BP_GO', 'MF_GO', 'CC_GO']:
        for n, pc in enumerate(list(kdtree_dict.keys())):
            print(n, ":  " + pc)
            print(go_class)
            kdtree_rez = kdtree_dict[pc][go_class]
            # kdtree_rez['GO ID'] = kdtree_rez['NNs_natlang'].map(go_map_dict) ## map_go_dict --> {GO_name: GO_ID}
            assoc = {}
            hash_gos = {i: go for i, go in enumerate(predterms_dict[pc][go_class]['combined']['GO ID'])}
            hash_gos_pos = {go: pos for go, pos in zip(predterms_dict[pc][go_class]['combined']['GO ID'],
                                                       predterms_dict[pc][go_class]['combined']['pos']
                                                       )}
            # hash_gos['dummy'] = 'dummy'
            pool = set()
            for i, go_id in hash_gos.items():
                # print(set(go_tree[go_id].get_all_children()))
                assoc[i] = set(go_tree[go_id].get_all_children())
                assoc[i].add(go_id)
                pool.update(list(assoc[i]))
            assoc['dummy'] = set(kdtree_rez['GO ID']) - pool
            ## cut kdtree results to iloc_cut to get 'sample population'
            pop_names = kdtree_rez.iloc[0:iloc_cut_dict[go_class]]
            pop_new = pop_names['GO ID']

            for ii in assoc:
                isSignif = False
                M = len(set(kdtree_rez['NNs_natlang']))  ## Total number of GO terms in MF, BP, or CC set
                n_hyper = len(set.intersection(assoc[ii], set(kdtree_rez
                                                              [
                                                                  'GO ID'])))  ## number of intersections between children terms of ML GO term of interest and full set of GO Terms
                N = len(set(pop_new))  ## Size of sample (should be equal to iloc_cut_dict[go_class])
                if not N == iloc_cut_dict[go_class]:
                    raise ValueError('N should be equal to iloc_cut. Currently N={} and iloc_cut_dict[{}]={}'
                                     '\nPlease check if you have used correct map_go_df or if drop_duplicates has messed up.'.format(
                        N, go_class, iloc_cut_dict[go_class]
                    ))
                successes = set.intersection(assoc[ii], set(pop_new))
                x = len(successes)  ## Number of successes
                pval = hypergeom.sf(x - 1, M, n_hyper, N)
                ## Bonferroni correction
                alpha_alt = pval / len(assoc.keys())
                # print(alpha_alt)
                alpha_crit = 1 - (1 - alpha_alt) ** (len(assoc.keys()))
                if alpha_crit < alpha_val:  ##Bonferroni correction for multiple testing
                    alpha_crit_str = str(alpha_crit) + '******'
                    isSignif = True
                else:
                    alpha_crit_str = str(alpha_crit)
                if not ii == 'dummy':
                    print('\t{}: {} {} alpha_crit = {}'.format(
                        ii, hash_gos[ii], go_tree[hash_gos[ii]].name, alpha_crit_str
                    )
                    )
                    print('\t\tM = {}; N = {}; n = {}; x = {}'.format(M, N, n_hyper, x))
                    res_dict[pc][go_class][hash_gos[ii]] = {'go_name': core.preprocess(go_tree[hash_gos[ii]].name),
                                                            'M': M, 'N': N, 'n_hyper': n_hyper, 'x': x, 'pval': pval,
                                                            'alpha_alt': alpha_alt, 'alpha_crit': alpha_crit,
                                                            'isSignif': isSignif,
                                                            'successes': successes,
                                                            'mapped': pop_names,
                                                            'pos': hash_gos_pos[hash_gos[ii]]
                                                            }
                else:
                    print('\t{} alpha_crit = {}'.format(ii, alpha_crit_str))
                    print('\t\tM = {}; N = {}; n = {}; x = {}'.format(M, N, n_hyper, x))
                    res_dict[pc][go_class][ii] = {'go_name': ii,
                                                  'M': M, 'N': N, 'n_hyper': n_hyper, 'x': x, 'pval': pval,
                                                  'alpha_alt': alpha_alt, 'alpha_crit': alpha_crit,
                                                  'isSignif': isSignif,
                                                  'successes': successes,
                                                  'mapped': pop_names,
                                                  'pos': np.nan
                                                  }
    return (res_dict)



