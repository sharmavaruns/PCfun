import os
import pandas as pd
import numpy as np
import sys
from pcfun import config
#import pcfun
from pcfun.config import fasttext_path
pcfun_path = '/Users/varunsharma/PycharmProjects/PCfun' ## please change path for you to COWs_model folder
sys.path.append(os.path.join(pcfun_path))


### This file should contain project/run specific configuration paths
root_path = '/Users/varunsharma/Documents/PCfun_stuff/Projects/Test1'
input_file_path = os.path.join(root_path,'input_df.tsv')

GO_class_oi = 'all' ## should be one of ('all','BP,'MF','CC')

input_df = pd.read_csv(input_file_path)
assert (input_df.columns == ['PC_names', 'PC_subunit_UniProts', 'PC_subunit_genenames']).all()

### Input DF --> PC vectors
#### Ideally should be something like:
#### pcfun.input_to_vecs(input_df,embed_path = pcfun.config.fasttext_path)
pc_names = list(input_df)