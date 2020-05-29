#!/usr/bin/env python3
"""Core utilities."""

# NOTE: here we should put core functionalities.
# In case of need we might add a folder to enforce a stricter hyerarchy.
# Let's discuss this.

import os
import numpy as np
import pandas as pd
from interact.nn_tree import NearestNeighborsTree
from interact import nn_tree
from interact import get_network_from_embedding_using_interact
import time
from interact.nn_tree import NeighborsMode
import textacy
# import pickle
# #from . import fasttext_path

##### Useful utility functions
preprocess = lambda x: textacy.preprocess.preprocess_text(x, lowercase=True, fix_unicode=True,no_punct=True)
class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value