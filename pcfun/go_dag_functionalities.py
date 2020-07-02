import os
import requests
import re
import networkx as nx
import itertools
import pygraphviz as pgv
import matplotlib
from pcfun.core import preprocess
import itertools
import pandas as pd
import scipy
import seaborn as sns
import time
import copy
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import fcluster


def go_graph_topchildren(go_dag, parent_term, recs, mapped_success_top10, nodecolor,
                         edgecolor, dpi,
                         draw_parents=True, draw_children=True):
    """Draw AMIGO style network, lineage containing one query record."""
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
                          dir="forward")  # , label="is_a")

    children = go_dag[parent_term].get_all_children()

    # recs_oi
    recs_oi = [go_dag[go_term_oi] for go_term_oi in mapped_success_top10['GO ID']]
    recs_oi_dict = {go_dag[go_term_oi]: score for go_term_oi, score in
                    zip(mapped_success_top10['GO ID'], mapped_success_top10['NNs_simil'])}

    cmap = matplotlib.cm.get_cmap('Blues')

    # rgba = cmap(0.5)
    # highlight the query terms
    val_col_map = {}
    for rec in recs:
        #print(rec.name)
        try:
            if rec in recs_oi:
                if rec.name == go_dag[parent_term].name:
                    val_col_map[rec.name] = matplotlib.colors.rgb2hex('plum')
                    #print('parent term: {}'.format(rec.id, rec.name), val_col_map[rec.name])
                    node = grph.get_node(go_dag.label_wrap(rec.item_id))
                    node.attr.update(fillcolor=val_col_map[rec.name])

                else:
                    #print(rec.id, rec.name)
                    # val_map[rec] = np.random.uniform(0,1)
                    # value = val_map.get(rec, recs_oi_dict[rec])
                    value = recs_oi_dict[rec]
                    val_col_map[rec.name] = matplotlib.colors.rgb2hex(cmap(recs_oi_dict[rec]))
                    # print(value)
                    node = grph.get_node(go_dag.label_wrap(rec.item_id))
                    node.attr.update(fillcolor=val_col_map[rec.name])
            elif rec.name == go_dag[parent_term].name:
                val_col_map[rec.name] = matplotlib.colors.rgb2hex('plum')
                #print('parent term: {}'.format(rec.id, rec.name), val_col_map[rec.name])
                node = grph.get_node(go_dag.label_wrap(rec.item_id))
                node.attr.update(fillcolor=val_col_map[rec.name])
        except:
            continue
    return grph, val_col_map


class GoGraph(nx.DiGraph):
    """Directed acyclic graph of Gene Ontology
    Attributes:
        alt_ids(dict): alternative IDs dictionary
        descriptors(set): flags and tokens that indicates the graph is
            specialized for some kind of analyses
        lower_bounds(collections.Counter):
            Pre-calculated lower bound count (Number of descendants + 1).
            Information content calculation requires precalc lower bounds.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alt_ids = {}  # Alternative IDs
        self.descriptors = set()
        self.lower_bounds = None
        # self.reversed = self.reverse(copy=False)

    def require(self, desc):
        if desc not in self.descriptors:
            raise exception.PGSSInvalidOperation(
                "'{}' is required.".format(desc))


def parse_block(lines):
    """Parse a Term block
    """
    term = {
        "alt_id": [],
        "relationship": []
    }
    splitkv = re.compile(r"(^[a-zA-Z_]+): (.+)")
    for line in lines:
        m = re.search(splitkv, line)
        # assert m, f"unexpected line: {line}"
        key = m.group(1)
        value = m.group(2)
        if key in ["id", "name", "namespace", "is_obsolete"]:
            term[key] = value
        elif key == "alt_id":
            term["alt_id"].append(value)
        elif key == "is_a":
            goid = value.split("!")[0].strip()
            term["relationship"].append({"type": "is_a", "id": goid})
        elif key == "relationship":
            typedef, goid = value.split("!")[0].strip().split(" ")
            term["relationship"].append({"type": typedef, "id": goid})
    # assert "id" in term, "missing id"
    # assert "name" in term, "missing name"
    # assert "namespace" in term, "missing namespace"
    return term


def blocks_iter(lines):
    """Iterate Term (and Typedef) blocks
    """
    type_ = None
    content = []
    termdef = re.compile(r"^\[([a-zA-Z_]+?)\]$")
    for line in lines:
        m = re.search(termdef, line)
        if m:
            if type_ is not None and content:
                yield {"type": type_, "content": content[:]}
            type_ = m.group(1)
            content.clear()
        elif line.rstrip():
            content.append(line.rstrip())
    if content:
        yield {"type": type_, "content": content[:]}


def from_obo_lines(lines, ignore_obsolete=True):
    lines_iter = iter(lines)

    # Header
    fv_line = next(lines_iter)
    format_ver = fv_line.split(":")[1].strip()
    # print(f"format-version: {format_ver}")

    # Build graph
    G = GoGraph()
    alt_ids = set()

    # Term blocks
    for tb in blocks_iter(lines_iter):
        if tb["type"] != "Term":
            # assert tb["type"] == "Typedef", f"unexpected type {tb['type']}"
            continue
        term = parse_block(tb["content"])

        # Ignore obsolete term
        obso = term.get("is_obsolete") == "true"
        if obso and ignore_obsolete:
            continue

        # Alternative ID mapping
        alt_ids |= set(term["alt_id"])
        for alt_id in term["alt_id"]:
            G.alt_ids[alt_id] = term["id"]

        # Add node
        attr = {
            "name": term["name"],
            "namespace": term["namespace"],
            "is_obsolete": obso
        }
        G.add_node(term["id"], **attr)
        for rel in term["relationship"]:
            G.add_edge(rel["id"], term["id"], type=rel["type"])

    # Check
    assert not (set(G) & alt_ids), "Inconsistent alternative IDs"
    assert len(G) >= 2, "The graph size is too small"
    assert G.number_of_edges(), "The graph has no edges"

    return G


def from_obo(pathlike, **kwargs):
    with open(pathlike, "rt") as f:
        G = from_obo_lines(f, **kwargs)
    return G


def makehash():
    """autovivification like hash in perl
     http://stackoverflow.com/questions/651794/whats-the-best-way-to-initialize-a-dict-of-dicts-in-python
     use call it on hash like h = makehash()
     then directly
     h[1][2]= 3
     useful ONLY for a 2 level hash
    """
    from collections import defaultdict
    return defaultdict(makehash)
    # return defaultdict(dict)


def map_retrieve(ids2map):
    '''
    Map database identifiers from/to UniProt accessions.
    '''
    base = "http://www.uniprot.org/uniprot/"
    end = "?query=id:" + ids2map
    add = "&format=tab&columns=go(cellular component),reviewed,protein names"
    response = requests.get(base + end + add)
    if response.ok:
        return parse_go_out(response.text)
    else:
        response.raise_for_status()


def parse_go_out(response):
    """
    grep GO:nr
    """
    cc = re.findall('\[GO:(\d+)\]', response)
    cc.sort()
    return ['GO:' + x for x in cc]


def read_gaf_out(go_path='tmp_GO_sp_only.txt'):
    """
    read gaf file and create a hash of hash
    gn => c
       => mf
       => bp
    """
    out = makehash()
    header = []
    temp = {}
    # go_path = io.resource_path('tmp_GO_sp_only.txt')
    go_path = go_path
    for line in open(go_path, mode='r'):
        line = line.rstrip('\n')
        if line.startswith(str('ID') + '\t'):
            header = re.split(r'\t+', line)
        else:
            things = re.split(r'\t+', line)
            temp = dict(zip(header, things))
        if len(temp.keys()) > 0:
            # assert False
            pr = str.upper(temp['GN'])
            for k in temp.keys():
                # if the key is the same
                if out[pr][k] and k is not 'ID' or 'GN':
                    out[pr][k] = ";".join([str(out[pr][k]), temp[k]])
                elif k is not 'ID' or 'GN':
                    out[pr][k] = temp[k]
    return out


def s_values(G, term):
    # wf = dict(zip(("is_a", 0.8), ("part_of", 0.6)))
    if not term in G:
        if term in G.alt_ids:
            term = G.alt_ids[term]
        else:
            raise ValueError(
                'It appears that {} does not exist in this GO Graph, nor in the alternative ids'.format(term))
    wf = dict(zip(("is_a", "part_of"), (0.8, 0.6)))
    sv = {term: 1}
    visited = set()
    level = {term}
    while level:
        visited |= level
        next_level = set()
        for n in level:
            for pred, edge in G.pred[n].items():
                weight = sv[n] * wf.get(edge["type"], 0)
                if pred not in sv:
                    sv[pred] = weight
                else:
                    sv[pred] = max([sv[pred], weight])
                if pred not in visited:
                    next_level.add(pred)
        level = next_level
    return {k: round(v, 3) for k, v in sv.items()}


def wang(G, term1, term2):
    """Semantic similarity based on Wang method
    Args:
        G(GoGraph): GoGraph object
        term1(str): GO term
        term2(str): GO term
        weight_factor(tuple): custom weight factor params
    Returns:
        float - Wang similarity value
    Raises:
        PGSSLookupError: The term was not found in GoGraph
    """
    #     if term1 not in G:
    #         return 0
    #         # raise Exception ("Missing term: " + term1)
    #     if term2 not in G:
    #         return 0
    #         # raise Exception ("Missing term: " + term2)
    sa = s_values(G, term1)
    sb = s_values(G, term2)
    sva = sum(sa.values())
    svb = sum(sb.values())
    common = set(sa.keys()) & set(sb.keys())
    cv = sum(sa[c] + sb[c] for c in common)
    return round(cv / (sva + svb), 3)


def parse_go(gn, gaf, go_type):
    """
    retrieve the GO gene names term from the
    """
    tmp = []
    try:
        tmp = gaf[gn][go_type].split(';')
    except AttributeError as e:
        tmp.append('NA')
    tmp = list(set(tmp))
    return [x for x in tmp if x is not 'NA']


def scr(G, gaf, id1, id2, go_type):
    """
    score using wang
    """
    t1 = parse_go(id1, gaf, go_type)
    t2 = parse_go(id2, gaf, go_type)
    if t1 and t2:
        x = [(wang(G, x[0], x[1])) for x in list(itertools.product(t1, t2))]
        return sum(x) / len(x)
    else:
        return 0


def combine_all2(G, gaf, t):
    """
    permute all of blocks of whatever
    """
    go_type = ['CC', 'MF', 'BP']
    out = []
    for go in go_type:
        x = [scr(G, gaf, x[0], x[1], go) for x in list(itertools.combinations(t, 2))]
        out.append(sum(x) / len(x))
    out.append(sum(out))
    return "\t".join([str(x) for x in out])


def common_parent_go_ids(terms, go):
    '''
        This function finds the common ancestors in the GO
        tree of the list of terms in the input.
        - input:
            - terms: list of GO IDs
            - go: the GO Tree object
        Taken from 'A Gene Ontology Tutorial in Python - Model Solutions to Exercises'
        by Alex Warwick
    '''
    # Find candidates from first
    rec = go[terms[0]]
    candidates = rec.get_all_parents()
    candidates.update({terms[0]})

    # Find intersection with second to nth term
    for term in terms[1:]:
        rec = go[term]
        parents = rec.get_all_parents()
        parents.update({term})

        # Find the intersection with the candidates, and update.
        candidates.intersection_update(parents)

    return candidates


def deepest_common_ancestor(terms, go):
    '''
        This function gets the nearest common ancestor
        using the above function.
        Only returns single most specific - assumes unique exists.
    '''
    # Take the element at maximum depth.
    return max(common_parent_go_ids(terms, go), key=lambda t: go[t].depth)


def lowest_common_ancestor(terms, go):
    '''
        This function gets the nearest common ancestor
        using the above function.
        Only returns single most specific - assumes unique exists.
    '''
    # Take the element at maximum depth.
    return min(common_parent_go_ids(terms, go), key=lambda t: go[t].depth)

class run_dca():
    def __init__(self,
                 path_obo:str,input_dat_path:str,go_dag,queries_vecs,queries_rez,go_map,nclusts: int = 25):
        import copy
        self.path_obo = path_obo
        self.input_dat_path = input_dat_path
        self.out_rez_path = os.path.join(os.path.dirname(self.input_dat_path), 'Results')
        os.makedirs(self.out_rez_path, exist_ok=True)
        self.go_dag = go_dag
        self.queries_vecs = queries_vecs
        self.queries_rez = queries_rez
        self.queries_rez_orig = copy.deepcopy(self.queries_rez)
        self.go_map = go_map
        self.G = from_obo(path_obo)
        self.nclusts = nclusts

    def parallelize_dataframe(df, func, n_cores=4):
        df_split = np.array_split(df, n_cores)
        pool = Pool(n_cores)
        df = pd.concat(pool.map(func, df_split))
        pool.close()
        pool.join()
        return df

    def semsim_squaredf(self,ids_ls,go_tree,go_dag):
        ids_ls_copy = copy.deepcopy(ids_ls)
        ids_ls_copy_df = pd.DataFrame(ids_ls_copy)
        ## mapping alternate GO IDs in order to calculate Wang SemSim
        ids_ls_copy_df = ids_ls_copy_df.replace(0,self.G.alt_ids)
        blank_df = pd.DataFrame(columns = list(ids_ls_copy_df[0]),index = list(ids_ls_copy_df[0]))
        id_names_zip_dict = {go_id:preprocess(go_dag[go_id].name) for go_id in ids_ls}
        #df_combos = pd.DataFrame(list(itertools.combinations_with_replacement(list(ids_ls_copy_df[0]), 2)))
        for x_run,y_run in itertools.combinations_with_replacement(list(ids_ls_copy_df[0]),2):
            # if not x in full_tree.nodes:
            #     x_run = full_tree.alt_ids[x]
            # else:
            #     x_run = x
            # if not y in full_tree.nodes:
            #     y_run = full_tree.alt_ids[y]
            # else:
            #     y_run = y
            #print(x_run,y_run)
            score = wang(go_tree,x_run,y_run)
            #print(x,y,score)
            blank_df[y_run].loc[x_run] = score
            blank_df[x_run].loc[y_run] = score
        blank_df.columns = ids_ls
        blank_df.index = ids_ls
        blank_df = blank_df.rename(columns=id_names_zip_dict,index=id_names_zip_dict)
        blank_df.columns.name = None
        blank_df.index.name = None
        blank_df = blank_df[blank_df.columns].astype(float)
        return(blank_df,id_names_zip_dict)

    def runner(self):
        print('start')
        # bp_tree = self.G.subgraph(
        #     [n for n, v in self.G.nodes(data=True) if v['namespace'] == 'biological_process'])
        # self.bp_tree = bp_tree
        n_clust = self.nclusts
        print('created BP Tree')
        for i, query in enumerate(list(self.queries_vecs.index)):  # queries_oi_names[:]):
            print(1,time.time())
            print(f'Functional Annotation Clustering for: {query}')
            k,id_names_zip_dict = self.semsim_squaredf(
                ids_ls=list(self.queries_rez[query]['BP_GO']['combined']['GO ID']),
                go_tree=self.G,
                go_dag=self.go_dag
            )
            print(2,time.time())
            Z = scipy.cluster.hierarchy.linkage(k, method='weighted',
                                                metric='euclidean')  ## calculate linkages for clusters
            print(3,time.time())
            clusters = fcluster(Z, n_clust, criterion='maxclust')
            print(4,time.time())
            clust_lists = []
            dcas = {}
            dcas_goids = {}
            for i in set(clusters):
                names = list(k.index[clusters == i])
                # print(names)
                clust_lists.append(names)
                go_ids = []
                for name in names:
                    go_ids.append(self.go_map['GO ID'][list(self.go_map['GO'] == name).index(True)])
                dca = deepest_common_ancestor(go_ids, self.go_dag)
                dcas[str(i)] = preprocess(self.go_dag[dca].name)
                dcas_goids[preprocess(self.go_dag[dca].name)] = dca
            print(5,time.time())
            go_term_clust_map = dict(zip(k.index, list(map(str, clusters))))
            clusters = \
                pd.DataFrame((list((k, dcas.get(v, v)) for (k, v) in go_term_clust_map.items()))).set_index(
                    [0])[1]
            print(6, time.time())
            row_colors = sns.color_palette("cubehelix", len(set(clusters)))
            lut = dict(zip(clusters.unique(), row_colors))
            clusters.name = None

            dca_clustermap = sns.clustermap(k, cmap='Blues', row_colors=clusters.map(lut), col_cluster=True,
                                            linewidths=0, xticklabels=False  # yticklabels=True,
                                            )
            for label in clusters.unique():
                dca_clustermap.ax_col_dendrogram.bar(0, 0, color=lut[label],
                                                     label=label, linewidth=0)
            dca_clustermap.fig.suptitle(query, ha='left', va='center').set_size(16)
            dca_clustermap.ax_col_dendrogram.legend(loc="lower left", ncol=3).set_title(
                'deepest common ancestor clusters', 'large')
            dca_clustermap.ax_col_dendrogram.set_xlim([0, 0])
            os.makedirs(os.path.join(self.out_rez_path,query),exist_ok=True)
            dca_clustermap.savefig(os.path.join(self.out_rez_path,query, 'FuncAnnotClust_DCA_plot.png'))
            print(7, time.time())
            ## Write out original ML predicted terms for each PC
            df_out = self.queries_rez_orig[query]['BP_GO']['combined']
            if not df_out.shape[0] == 0:
                df_out.to_csv(
                    os.path.join(self.out_rez_path,query,'ML_pred_results_before_DCA_func_clustering.tsv'),
                    sep = '\t'
                )
            else:
                print(query,
                      'has no ML predicted terms for BP_GO, hence no reason to write out original ML predicted'
                      ' terms prior to doing functional annotation clustering based on pairwise semantic similarity.'
                      )
            print(8, time.time())
            ## Overwrite DCA GO terms to self.queries_rez
            self.queries_rez[query]['BP_GO']['combined'] = pd.DataFrame(
                dcas_goids, index=['GO ID']).T.drop_duplicates()
            print(9, time.time())
        return(self.queries_rez,self.queries_rez_orig)
