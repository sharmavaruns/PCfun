import pandas as pd
import os
from pcfun import mapping as mpng

home_dir = os.path.expanduser('~')

path_corum = os.path.join(home_dir,'pcfun/coreComplexes.txt')
# parse file
corum_complexes_df = pd.read_csv(
    path_corum, sep='\t', index_col=0
)
corum_complexes_cut = corum_complexes_df[['ComplexName','Organism','subunits(UniProt IDs)','GO ID']]
corum_complexes_cut['subunits(UniProt IDs)'] = [str_.split(';') for str_ in corum_complexes_cut['subunits(UniProt IDs)']]
corum_complexes_cut = corum_complexes_cut.loc[corum_complexes_cut['subunits(UniProt IDs)'].apply(lambda x: x[0] != 'None')]
corum_complexes_cut['subunits(UniProt IDs)'] = corum_complexes_cut['subunits(UniProt IDs)'].apply(lambda x: list(set(x)))

df_process = corum_complexes_cut[['ComplexName','subunits(UniProt IDs)']]
df_process.columns = ['ComplexName','UniProtIDs']
df_process = df_process.reset_index(drop=True)
df_process['UniProtIDs'] = df_process['UniProtIDs'].apply(lambda x: ';'.join(x))

embed_model = mpng.ftxt_model(path_to_fasttext_embedding=f'{home_dir}/pcfun/Embeddings/fulltext_model.bin',
                              req_inputs_path=f'{home_dir}/pcfun/')

#### Map the ';' delimited UniProt Subunits to their Gene Names
gn_df,uni_df,dropped_idxs = embed_model.uniprots_to_gnames(pd.DataFrame(list(df_process['UniProtIDs'])))
### Remember that above line mapping uniprot ids to gene names drops rows where a subunit maps to 'OBSOLETE' or 'UnmappedUniProtID'

df_process_new = df_process.drop(dropped_idxs).reset_index(drop=True)
df_process_new['GeneNames'] = gn_df[0]

## Write entire mapping file to Toy_Data_Input folder for people's reference
#### Has Original 'ComplexName', 'UniProtIDs', and 'GeneNames' mapped per complex
df_process_new.to_csv('/Users/varunsharma/PycharmProjects/PCfun/Toy_Data_Input/corum_complexes_names_mapped.tsv',sep='\t')

## Write out top 50 complexes for 'ComplexName', 'UniProtIDs', or 'GeneNames' queries
df_process_new['ComplexName'].iloc[:50].to_csv('/Users/varunsharma/PycharmProjects/PCfun/Toy_Data_Input/input_df-FullComplexNames.tsv',
                      sep='\t',header=False,index=False)
df_process_new['UniProtIDs'].iloc[:50].to_csv('/Users/varunsharma/PycharmProjects/PCfun/Toy_Data_Input/input_df-UniProtIDs.tsv',
                      sep='\t',header=False,index=False)
df_process_new['GeneNames'].iloc[:50].to_csv('/Users/varunsharma/PycharmProjects/PCfun/Toy_Data_Input/input_df-GeneNames.tsv',
                      sep='\t',header=False,index=False)