#!/usr/bin/env python

### Objective of this script is to read-in .tsv file with columns corresponding to the PC info

###### Assuming that the input is only a single column .tsv file with the input names
###### Can add is_UniProt to function to first map uniprot ids to gene names if necessary
###### Will have to implement UniProt ID --> Gene Name script (which is a pain) so deferring that to a little later

import pandas as pd
import numpy as np
import fasttext
import textacy
import os
from pcfun.core import preprocess
import time
import urllib.parse
import urllib.request
import io


class ftxt_model():
    def __init__(self, path_to_fasttext_embedding: str, req_inputs_path:str):
        self.model = fasttext.load_model(path_to_fasttext_embedding)
        self.supp_tax_ids = [9606, 3702, 6239]  ## Should make this modular to look up ftp link and extract tax_ids
        self.req_inputs_path = req_inputs_path
        print('Model should be loaded')

    def queries_df_to_vecs(self,input_df):
        if isinstance(input_df,list):
            input_df = pd.DataFrame(input_df)
        if input_df.shape[1] != 1:
            raise ValueError(
                f'Expected input pd.DataFrame to have single column with no header of natural language queries'
                f'Can also input a list of queries that will be coerced into a pd.DataFrame'
            )
        ## Preprocess strings
        input_df[0] = input_df[0].apply(lambda x: preprocess(str(x)))
        ## drop duplicates
        input_df = input_df.drop_duplicates(keep='first')
        ## get embedding sentence vectors for queries
        vecs_df = pd.DataFrame(list(input_df[0].apply(self.model.get_sentence_vector)), index=input_df[0])
        ## L2 normalize vectors
        vec_norm = np.sqrt(np.square(np.array(vecs_df)).sum(axis=1))
        queries_vec_normalized = pd.DataFrame(np.array(vecs_df) / vec_norm.reshape(vecs_df.shape[0], 1),
                                              index=vecs_df.index)
        return(queries_vec_normalized)

    # def uniprots_to_gnames(self,input_df):
    #     input_df_split = input_df[0].apply(lambda x: x.split(';'))
    #     queries_ls = list(input_df_split)
    #     uni_ids_set = set([subls for ls in queries_ls for subls in ls])
    #     query_ls = list(uni_ids_set)
    #     url = 'https://www.uniprot.org/uploadlists/'
    #
    #     params = {
    #         'from': 'ACC+ID',
    #         'to': 'GENENAME',
    #         'format': 'tab',
    #         'query': ' '.join(query_ls)  # 'P40925 P40926 O43175 Q9UM73 P97793'
    #     }
    #     data = urllib.parse.urlencode(params)
    #     data = data.encode('utf-8')
    #     req = urllib.request.Request(url, data)
    #     with urllib.request.urlopen(req) as f:
    #         response = f.read()
    #     # print(response.decode('utf-8'))
    #     df_map = pd.read_csv(io.StringIO(response.decode('utf-8')), sep='\t')
    #     #return(df_map,0)
    #     df_map.columns = ['UniProt_ID', 'Gene_Name']
    #     print('hello')
    #     uni_gn_dict = dict(zip(df_map['UniProt_ID'],df_map['Gene_Name']))
    #
    #     #test_ls_split = list(map(lambda xx: xx.split(';'), queries_ls))
    #     #uni_gn_dict = {'a': '1', 'b': '2', 'c': '3', 'd': '4', 'l': '5', 'j': '6'}
    #     return(queries_ls,uni_gn_dict,df_map)
    #     ########## Getting a none type because a particular uniprot is not in the mapped set.
    #     ########## use: input_df = pd.DataFrame({0:list(corum_complexes_cut['subunits(UniProt IDs)'].apply(lambda x: ';'.join(x)))})
    #     ########## use: queries_ls,uni_gn_dict,df_map = uniprots_to_gnames(input_df)
    #     ########## use: uni_ids_set = set([subls for ls in queries_ls for subls in ls])
    #     ########## use: uni_ids_set - uni_gn_dict.keys() to for example test set of issues
    #     repl_ls = [';'.join(list(map(uni_gn_dict.get, keys))) for keys in queries_ls]
    #     input_df_old = input_df.copy(deep=True)
    #     input_df = pd.DataFrame(repl_ls)
    #     # repl_ls == [['1', '2', '3'], ['2', '5', '4', '6']]
    #     return(input_df,input_df_old)

    def tsv_to_vecs(self, path_to_tsv: str, is_UniProt=False, write_vecs=True, **kwargs):
        '''

        :param path_to_tsv {str}: Path navigating to .tsv file with single column and no header of input pc queries
            if input queries are subunits (either Gene Names or UniProt IDs), they should be ";" delimited
            if input UniProt ids of subunits, make sure to set is_UniProt=True and supply Taxonomic ID for file
                otherwise defaults to assuming 9606 for human taxonomy id to attempt UniProt --> Gene Name map
        :param is_UniProt {bool}: Boolean indicating if input queries in .tsv file are UniProt IDs for PC subunits
            if True, assumes input subunit ids are UniProt ids delimited by ";". Ensure to include optional taxon_id
                number to allow function to download relevant mapping file from UniProt and
                do UniProt --> Gene Name mapping
        :return:
        '''
        input_df = pd.read_csv(path_to_tsv, sep='\t', header=None)
        if input_df.shape[1] != 1:
            raise ValueError(
                f'Expected input .tsv file to have single column with no header\n'
                f'Check input file at: {path_to_tsv}'
            )
        if is_UniProt:
            if kwargs.get('taxon_id', None) == None:
                import warnings
                warnings.warn(
                    'You have set True for "is_UniProt" without providing "taxon_id" argument. Therefore '
                    'defaulting to taxon_id=9606 for homo sapiens. Will attempt to download that file '
                    'to map UniProt IDs to Gene Names'
                )
            taxon_id = kwargs.get('taxon_id', 9606)
            try:
                taxon_id = int(taxon_id)
                if not taxon_id in set(self.supp_tax_ids):
                    raise ValueError(
                        f'You have input an unsupported taxon id. '
                        f'Please input an integer corresponding to one of the following taxon_ids that '
                        f'are supported by UniProt. I am downloading files from: '
                        f'ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/'
                        f'\nSupported taxon ids are: {self.supp_tax_ids}'
                    )
            except:
                raise ValueError(
                    f'You have input a taxon id that is not able to be converted to an integer. '
                    f'Please input an integer corresponding to one of the following taxon_ids that '
                    f'are supported by UniProt. I am downloading files from: '
                    f'ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/'
                    f'\nSupported taxon ids are: {self.supp_tax_ids}'
                )

            raise ValueError(f'UniProt ID mapping to GeneName not yet implemented')
        queries_vec_normalized = self.queries_df_to_vecs(input_df=input_df)
        if write_vecs:
            prefix = kwargs.get('vecs_file_prefix',f'out_{time.time()}')
            out_name = prefix + '_vecs.tsv'
            print('Writing out vecs\n',os.path.join(os.path.dirname(path_to_tsv), out_name))
            queries_vec_normalized.to_csv(
                os.path.join(os.path.dirname(path_to_tsv), out_name),
                sep='\t', header=True, index=True
            )
        return (queries_vec_normalized)
    # def uniprot_to_genename_map(self,taxon_id: int):
    #     map_uni_gn_pn_spec = []
    #     with open(os.path.join(uniprot_path, 'uniprot_taxo=9606_human_20191112.fasta'), 'r') as f:
    #         for i, line in enumerate(f):
    #             if '>' in line:
    #                 split_line = line.split('|')
    #                 uni = split_line[1]
    #                 try:
    #                     try:
    #                         gn = re.search(r'(?<=GN=)(.*?)(?= )', split_line[2]).group()
    #                     except AttributeError:
    #                         gn = 'uncharacterized'
    #                     try:
    #                         protname = re.search(r'(?<= )(.*?)(?= OS)', split_line[2]).group()
    #                     except AttributeError:
    #                         protname = 'uncharacterized'
    #                     try:
    #                         species = re.search(r'(?<= OS=)(.*?)(?= OX=)', split_line[2]).group()
    #                     except AttributeError:
    #                         species = 'uncharacterized'
    #                     map_uni_gn_pn_spec.append([i, uni, gn, protname, species])
    #                 except:
    #                     print('line {} has something missing\n{}'.format(i, line))
    #     #             if i > 1000:
    #     #                 break
    #     map_uni_gn_pn_spec_df = pd.DataFrame(map_uni_gn_pn_spec,
    #                                          columns=['idx_in_file', 'Uniprot', 'GeneName', 'ProtName', 'Species'])
    #     # map_uni_gn_pn_spec_df.to_csv(os.path.join(uniprot_path,'taxo=9606__human__uni_gn_pn_spec'),sep='\t')
    #     map_uni_gn_pn_spec_df

### Example Usage (Will be moved to main.py script when ready)
# fasttext_path = '/Users/varunsharma/Documents/PCfun_stuff/req_inputs/Embeddings/abstracts_model.bin'
# input_dat_path = '/Users/varunsharma/Documents/PCfun_stuff/Projects/Test1/input_df.tsv'
# abstr_model = ftxt_model(path_to_fasttext_embedding=fasttext_path)
# test_vecs = abstr_model.tsv_to_vecs(path_to_tsv=input_dat_path,write_vecs=True)
# time_start = time.time()
# path_corum = '/Users/varunsharma/Documents/PCfun_stuff/req_inputs/coreComplexes.txt'
# # parse file
# corum_complexes_df = pd.read_csv(
#     path_corum, sep='\t', index_col=0
# )
# corum_complexes_cut = corum_complexes_df[['ComplexName','Organism','subunits(UniProt IDs)','GO ID']]
# corum_complexes_cut['subunits(UniProt IDs)'] = [str_.split(';') for str_ in corum_complexes_cut['subunits(UniProt IDs)']]
# corum_complexes_cut = corum_complexes_cut.loc[corum_complexes_cut['subunits(UniProt IDs)'].apply(lambda x: x[0] != 'None')]
# corum_complexes_cut['subunits(UniProt IDs)'] = corum_complexes_cut['subunits(UniProt IDs)'].apply(lambda x: list(set(x)))
#
#
#
# time_end = time.time()
# print(f'Time taken: {(time_end-time_start)/60:0.2f} min')