# /bin/python2.7
import os
import sys
import numpy as np
sys.path.append('../matchzoo/inputs')
sys.path.append('../matchzoo/utils')
from preparation import *
from preprocess import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',  help='Data path for the generated files.')
    args = parser.parse_args()

    basedir = args.data_path #'../../data/toy_example/ranking/'

    # transform query/document pairs into corpus file and relation file
    prepare = Preparation()
    corpus, rels = prepare.run_with_one_corpus(os.path.join( basedir, 'sample.txt'))
    print('total corpus : %d ...' % (len(corpus)))
    print('total relations : %d ...' % (len(rels)))
    prepare.save_corpus(os.path.join(basedir, 'corpus.txt'), corpus)

    rel_train, rel_valid, rel_test = prepare.split_train_valid_test_for_ranking(rels, [0.4, 0.3, 0.3])
    prepare.save_relation(os.path.join(basedir, 'relation_train.txt'), rel_train)
    prepare.save_relation(os.path.join(basedir, 'relation_valid.txt'), rel_valid)
    prepare.save_relation(os.path.join(basedir, 'relation_test.txt'), rel_test)
    print('preparation finished ...')

    # Prerpocess corpus file
    preprocessor = Preprocess()

    dids, docs = preprocessor.run(os.path.join(basedir, 'corpus.txt'))
    preprocessor.save_word_dict(os.path.join(basedir, 'word_dict.txt'))
    preprocessor.save_words_stats(os.path.join(basedir,'word_stats.txt'))

    fout = open(os.path.join(basedir, 'corpus_preprocessed.txt'),'w')
    for inum,did in enumerate(dids):
        fout.write('%s %s %s\n' % (did, len(docs[inum]), ' '.join(map(str, docs[inum]))))
    fout.close()
    print('preprocess finished ...')

