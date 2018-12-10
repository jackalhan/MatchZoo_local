# /bin/python2.7
import os
import sys
import numpy as np
sys.path.append('../matchzoo/inputs')
sys.path.append('../matchzoo/utils')
from preparation import *
from preprocess import *
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',  help='Data path for the generated files.')
    parser.add_argument('--is_split_from_one_file', default=True, type=str2bool, help='Train raw file')
    parser.add_argument('--train_file', default=None,type=str, help='Train or base raw file')
    parser.add_argument('--test_file', default=None, type=str,help='Test raw file')
    parser.add_argument('--eval_file', default=None, type=str,help='Eval raw file')
    args = parser.parse_args()

    basedir = args.data_path #'../../data/toy_example/ranking/'

    # transform query/document pairs into corpus file and relation file
    prepare = Preparation()
    if args.is_split_from_one_file:
        corpus, rels = prepare.run_with_one_corpus(os.path.join( basedir, args.train_file))
        print('total corpus : %d ...' % (len(corpus)))
        print('total relations : %d ...' % (len(rels)))
        rel_train, rel_valid, rel_test = prepare.split_train_valid_test_for_ranking(rels, [0.4, 0.3, 0.3])

    else:
        infiles = [os.path.join(basedir, args.train_file),
                   os.path.join(basedir, args.test_file),
                   os.path.join(basedir, args.eval_file if args.eval_file is not None else args.test_file) ]
        corpus, rel_train, rel_valid, rel_test = prepare.run_with_train_valid_test_corpus(infiles[0], infiles[1],
                                                                                          infiles[2])
    print('total corpus : %d ...' % (len(corpus)))
    print('total relation-train : %d ...' % (len(rel_train)))
    print('total relation-valid : %d ...' % (len(rel_valid)))
    print('total relation-test: %d ...' % (len(rel_test)))

    prepare.save_corpus(os.path.join(basedir, 'corpus.txt'), corpus)
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

