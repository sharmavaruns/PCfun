#!/usr/bin/env python3

"""Core utilities."""
import textacy

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