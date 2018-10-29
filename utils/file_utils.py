# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 11:58:26 2018

file_utils

@author: GEO
"""


def print_and_save(text, path):
    print(text)
    if path is not None:
        with open(path, 'a') as f:
            print(text, file=f)
