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


class ftxt_model():
    def __init__(self, path_to_fasttext_model: str):
        self.model = fasttext.load_model(path_to_fasttext_model)
        self.supp_tax_ids = [9606, 3702, 6239]  ## Should make this modular to look up ftp link and extract tax_ids
        print('Model should be loaded')

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

        input_df[0] = input_df[0].apply(lambda x: preprocess(str(x)))
        input_df = input_df.drop_duplicates(keep='first')
        vecs_df = pd.DataFrame(list(input_df[0].apply(self.model.get_sentence_vector)), index=input_df[0])
        ## L2 normalize vectors
        vec_norm = np.sqrt(np.square(np.array(vecs_df)).sum(axis=1))
        queries_vec_normalized = pd.DataFrame(np.array(vecs_df) / vec_norm.reshape(vecs_df.shape[0], 1),
                                              index=vecs_df.index)
        if write_vecs:
            queries_vec_normalized.to_csv(
                os.path.join(os.path.dirname(path_to_tsv), 'vecs.tsv'),
                sep='\t', header=True, index=True
            )
        return (queries_vec_normalized)


### Example Usage (Will be moved to main.py script when ready)
fasttext_path = '/Users/varunsharma/Documents/PCfun_stuff/req_inputs/Embeddings/abstracts_model.bin'
input_dat_path = '/Users/varunsharma/Documents/PCfun_stuff/Projects/Test1/input_df.tsv'
abstr_model = ftxt_model(path_to_fasttext_model=fasttext_path)
test_vecs = abstr_model.tsv_to_vecs(path_to_tsv=input_dat_path,write_vecs=True)
