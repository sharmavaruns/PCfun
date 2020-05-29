#!/usr/bin/env python

### Proposed to have this created by another script called 'make_config.py'
### I'd like to have a config file that has paths and stuff recorded for user and callable
### by module rather than as global variables
import os


root_path = os.path.join('/Users/varunsharma/Documents/PCfun_stuff')
req_input_path = os.path.join(root_path,'req_inputs')
embedding_dir_path = os.path.join(req_input_path,'Embeddings')
fasttext_path = os.path.join(req_input_path,'fastText-0.2.0')
superv_uni_fmodels_path = os.path.join(req_input_path,'PC_GO_Uniprot','full_models')
superv_cmplname_fmodels_path = os.path.join(req_input_path,'PC_GO_w_complex','full_models')
trees_path = os.path.join(req_input_path,'Trees_NNs')
additional_trees = os.path.join(trees_path,'additional_trees')
custom_trees = os.path.join(trees_path,'customTrees')
go_obo_path = os.path.join(req_input_path,'go-basic.obo')
