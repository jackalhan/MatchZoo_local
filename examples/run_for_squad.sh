#!/bin/bash
path="/home/jackalhan/Development/github/MatchZoo_local/examples/squad/dev_no_lower_no_filter/"
train_file="squad_to_matchzoo_train.txt"
test_file="squad_to_matchzoo_dev.txt"
eval_file="squad_to_matchzoo_dev.txt"
glove="${path}glove.840B.300d.txt"
word_dict="${path}word_dict.txt"
embed_glove="${path}embed_glove_d300"
normed_glove="${path}embed_glove_d300_norm"

#python test_preparation_for_ranking.py --data_path=${path} --is_split_from_one_file=False --train_file=${train_file} --test_file=${test_file} --eval_file=${eval_file}

python gen_w2v.py ${glove} ${word_dict} ${embed_glove}

python norm_embed.py ${embed_glove} ${normed_glove}

cd squad/dev_no_lower_no_filter/
cat word_stats.txt | cut -d ' ' -f 1,4 > embed.idf
cd ..
cd ..

python gen_hist4drmm.py 60 ${path}
