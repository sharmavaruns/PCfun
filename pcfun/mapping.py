#!/usr/bin/env python

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

    def uniprots_to_gnames(self,input_df):
        '''

        :param input_df: a single column .tsv file with the input UniProt Subunit IDs delimiited by ';'
        :return: 3 dfs:
            - input_df --> newly mapped GeneNames delimited by ';' with rows dropped where a subunit maps to 'OBSOLETE' or 'UnmappedUniProtID'
            - input_df_old --> the original input df with the UniProt Subunit IDs delimited by ';'
            - dropped_queries --> list of integer indexes corresponding to the rows that were dropped in 'input_df_old'
        '''
        input_df_split = input_df[0].apply(lambda x: x.split(';'))
        queries_ls = list(input_df_split)
        uni_ids_set = set([subls for ls in queries_ls for subls in ls])
        query_ls = list(uni_ids_set)
        url = 'https://www.uniprot.org/uploadlists/'

        params = {
            'from': 'ACC+ID',
            'to': 'GENENAME',
            'format': 'tab',
            'query': ' '.join(query_ls)  # e.g. 'P40925 P40926 O43175 Q9UM73 P97793'
        }
        data = urllib.parse.urlencode(params)
        data = data.encode('utf-8')
        req = urllib.request.Request(url, data)
        with urllib.request.urlopen(req) as f:
            response = f.read()
        df_map = pd.read_csv(io.StringIO(response.decode('utf-8')), sep='\t')
        uni_gn_dict = dict(zip(df_map['From'], df_map['To']))

        ### For UniProt IDs where no gene names were returned, mapping to Gene ID
        no_gns = list(uni_ids_set - uni_gn_dict.keys())
        params = {
            'from': 'ACC+ID',
            'to': 'ID',
            'format': 'tab',
            'query': ' '.join(no_gns)
        }
        data = urllib.parse.urlencode(params)
        data = data.encode('utf-8')
        req = urllib.request.Request(url, data)
        with urllib.request.urlopen(req) as f:
            response = f.read()

        df_map_no_gns = pd.read_csv(io.StringIO(response.decode('utf-8')), sep='\t')
        df_map_no_gns['To'] = df_map_no_gns['To'].apply(lambda x: x.split('_')[0])

        df_map_final = pd.concat([df_map,df_map_no_gns],ignore_index=True)

        ## keeping first returned Gene Name for UniProt IDs with multiple gene names returned
        df_map_final = df_map_final.drop_duplicates('From',keep='first').reset_index(drop=True)
        uni_gn_dict_final = dict(zip(df_map_final['From'], df_map_final['To']))
        no_gn_final = list(uni_ids_set - uni_gn_dict_final.keys())
        df_map_final = pd.concat([
            df_map_final,
            pd.DataFrame({'From':no_gn_final,'To':['UnmappedUniProtID']*len(no_gn_final)})
        ],ignore_index=True)
        uni_gn_dict_final = dict(zip(df_map_final['From'], df_map_final['To']))

        assert (len(uni_ids_set - set(df_map_final['From'])) == 0)

        repl_ls = [] # list to store replaced queries with UniProt Mapped IDs
        dropped_queries = []
        for idx,uni_query in enumerate(queries_ls):
            mapped_rez = list(map(uni_gn_dict_final.get, uni_query))
            if 'OBSOLETE' in mapped_rez or 'UnmappedUniProtID' in mapped_rez:
                print(
                    f'query # {idx}: {uni_query} maps to --> {mapped_rez}.\n'
                    f'Since some subunits are mapped to obsolete or are Unmapped, this query will be removed. '
                )
                dropped_queries.append(idx)
            else:
                repl_ls.append(';'.join(mapped_rez))

        input_df_old = input_df.copy(deep=True)
        input_df = pd.DataFrame(repl_ls)
        return(input_df,input_df_old,dropped_queries)

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
        print('Starting conversion process from queries to vecs.')
        input_df = pd.read_csv(path_to_tsv, sep='\t', header=None)
        if input_df.shape[1] != 1:
            raise ValueError(
                f'Expected input .tsv file to have single column with no header\n'
                f'Check input file at: {path_to_tsv}'
            )
        print(f'is_UniProt = {is_UniProt}')
        if is_UniProt:
            print('Mapping UniProt IDs in queries to Gene Names.')
            input_df, input_df_old, dropped_queries = self.uniprots_to_gnames(input_df)
            input_df.to_csv(path_to_tsv,index=False,header=False,sep = '\t')
            path_to_old_input_df = path_to_tsv.split('.')[0]+'_preUniProtmapped.tsv'
            input_df_old.to_csv(path_to_tsv, index=False, header=False, sep='\t')

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