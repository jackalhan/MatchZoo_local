#!/usr/bin/env python
# coding: utf-8
from __future__ import  print_function

import os
import sys
import random
random.seed(49999)
import numpy
numpy.random.seed(49999)

sys.path.append('../matchzoo/inputs/')
sys.path.append('../matchzoo/utils/')

from preparation import Preparation
from preprocess import Preprocess, NgramUtil


def read_dict(infile):
    word_dict = {}
    for line in open(infile):
        r = line.strip().split()
        word_dict[r[1]] = r[0]
    return word_dict


def read_doc(infile):
    doc = {}
    for line in open(infile):
        r = line.strip().split()
        doc[r[0]] = r[1:]
        #assert len(doc[r[0]]) == int(r[1])
    return doc


def filter_triletter(tri_stats, min_filter_num=5, max_filter_num=10000):
    tri_dict = {}
    tri_stats = sorted(tri_stats.items(), key=lambda d:d[1], reverse=True)
    for triinfo in tri_stats:
        if min_filter_num <= triinfo[1] <= max_filter_num:
            if triinfo[0] not in tri_dict:
                tri_dict[triinfo[0]] = len(tri_dict)
    return tri_dict


if __name__ == '__main__':
    prepare = Preparation()
    srcdir = sys.argv[2]
    dstdir = srcdir

    word_dict_input = dstdir + 'word_dict.txt'
    triletter_dict_output = dstdir + 'triletter_dict.txt'
    word_triletter_output = dstdir + 'word_triletter_map.txt'
    word_dict = read_dict(word_dict_input)
    word_triletter_map = {}
    triletter_stats = {}
    for wid, word in word_dict.items():
        nword = '#' + word + '#'
        ngrams = NgramUtil.ngrams(list(nword), 3, '')
        word_triletter_map[wid] = []
        for tric in ngrams:
            if tric not in triletter_stats:
                triletter_stats[tric] = 0
            triletter_stats[tric] += 1
            word_triletter_map[wid].append(tric)
    triletter_dict = filter_triletter(triletter_stats, 5, 10000)
    with open(triletter_dict_output, 'w') as f:
        for tri_id, tric in triletter_dict.items():
            print(tri_id, tric, file=f)
    with open(word_triletter_output, 'w') as f:
        for wid, trics in word_triletter_map.items():
            print(wid, ' '.join([str(triletter_dict[k]) for k in trics if k in triletter_dict]), file=f)

    print('Triletter Processing finished ...')

