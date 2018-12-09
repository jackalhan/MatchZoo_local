# -*- coding: utf8 -*-
from __future__ import print_function
import os
import sys
import time
import json
import argparse
import random
random.seed(49999)
import numpy
numpy.random.seed(49999)
import tensorflow
tensorflow.set_random_seed(49999)
import matplotlib.pyplot as plt
from collections import OrderedDict
import atexit

import keras
import keras.backend as K
from keras.models import Sequential, Model
#from keras.callbacks import TensorBoard

from utils import *
import inputs
import metrics
from losses import *
from optimizers import *

config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
sess = tensorflow.Session(config = config)
stats_for_plots = dict()
logs_dir = None
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.eager import context

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', global_step=0, **kwargs):
        self.val_log_dir = os.path.join(log_dir, 'validation')
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)
        self.global_step = global_step
    def set_model(self, model):
        if context.executing_eagerly():
            self.val_writer = tf.contrib.summary.create_file_writer(self.val_log_dir)
        else:
            self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def _write_custom_summaries(self, step, logs=None):

        #if self._total_batches_seen == 0:
        step += self.global_step
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if 'val_' in k}
        # print("Step: {}".format(step))
        #print("Current Total Batch Seen/Step For TRAIN: {}/{}".format( self._total_batches_seen, step))

        if context.executing_eagerly():
            with self.val_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for name, value in val_logs.items():
                    tf.contrib.summary.scalar(name, value.item(), step=step)
        else:
            for name, value in val_logs.items():
                #print("Current Total Batch Seen/Step For TEST: {}/{}".format(self._total_batches_seen, step))
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.val_writer.add_summary(summary, step)
        self.val_writer.flush()

        logs = {k: v for k, v in logs.items() if not 'val_' in k}
        super(TrainValTensorBoard, self)._write_custom_summaries(step, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

def load_model(config):
    global_conf = config["global"]
    model_type = global_conf['model_type']
    if model_type == 'JSON':
        mo = Model.from_config(config['model'])
    elif model_type == 'PY':
        model_config = config['model']['setting']
        model_config.update(config['inputs']['share'])
        sys.path.insert(0, config['model']['model_path'])

        model = import_object(config['model']['model_py'], model_config)
        mo = model.build()
    return mo


def train(config):

    print(json.dumps(config, indent=2), end='\n')
    # read basic config
    global_conf = config["global"]
    optimizer = global_conf['optimizer']
    optimizer=optimizers.get(optimizer)
    K.set_value(optimizer.lr, global_conf['learning_rate'])
    weights_file = str(global_conf['weights_file']) + '.%d'

    global logs_dir
    logs_dir = str(global_conf['logs'])
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    display_interval = int(global_conf['display_interval'])
    num_iters = int(global_conf['num_iters'])
    save_weights_iters = int(global_conf['save_weights_iters'])

    # read input config
    input_conf = config['inputs']
    share_input_conf = input_conf['share']


    # collect embedding
    if 'embed_path' in share_input_conf:
        embed_dict = read_embedding(filename=share_input_conf['embed_path'])
        _PAD_ = share_input_conf['vocab_size'] - 1
        embed_dict[_PAD_] = np.zeros((share_input_conf['embed_size'], ), dtype=np.float32)
        embed = np.float32(np.random.uniform(-0.2, 0.2, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = convert_embed_2_numpy(embed_dict, embed = embed)
    else:
        embed = np.float32(np.random.uniform(-0.2, 0.2, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = embed
    print('[Embedding] Embedding Load Done.', end='\n')

    # list all input tags and construct tags config
    input_train_conf = OrderedDict()
    input_eval_conf = OrderedDict()
    input_eval_loss_conf = OrderedDict()
    for tag in input_conf.keys():
        if 'phase' not in input_conf[tag]:
            continue
        if input_conf[tag]['phase'] == 'TRAIN':
            input_train_conf[tag] = {}
            input_train_conf[tag].update(share_input_conf)
            input_train_conf[tag].update(input_conf[tag])
            stats_for_plots[tag + '_loss'] = dict()
        elif input_conf[tag]['phase'] == 'EVAL':
            input_eval_conf[tag] = {}
            input_eval_conf[tag].update(share_input_conf)
            input_eval_conf[tag].update(input_conf[tag])
            stats_for_plots[tag] = dict()
        elif input_conf[tag]['phase'] == 'EVAL_LOSS':
            input_eval_loss_conf[tag] = {}
            input_eval_loss_conf[tag].update(share_input_conf)
            input_eval_loss_conf[tag].update(input_conf[tag])
            stats_for_plots[tag] = dict()
    print('[Input] Process Input Tags. %s in TRAIN, %s in EVAL, %s in EVAL_LOSS.' % (input_train_conf.keys(), input_eval_conf.keys(), input_eval_loss_conf.keys()), end='\n')

    # collect dataset identification
    dataset = {}
    for tag in input_conf:
        if tag != 'share' and input_conf[tag]['phase'] == 'PREDICT':
            continue
        if 'text1_corpus' in input_conf[tag]:
            datapath = input_conf[tag]['text1_corpus']
            if datapath not in dataset:
                dataset[datapath], _ = read_data(datapath)
        if 'text2_corpus' in input_conf[tag]:
            datapath = input_conf[tag]['text2_corpus']
            if datapath not in dataset:
                dataset[datapath], _ = read_data(datapath)
    print('[Dataset] %s Dataset Load Done.' % len(dataset), end='\n')

    # initial data generator
    train_gen = OrderedDict()
    eval_gen = OrderedDict()
    eval_loss_gen = OrderedDict()

    for tag, conf in input_train_conf.items():
        print(conf, end='\n')
        conf['data1'] = dataset[conf['text1_corpus']]
        conf['data2'] = dataset[conf['text2_corpus']]
        generator = inputs.get(conf['input_type'])
        train_gen[tag] = generator( config = conf )

    for tag, conf in input_eval_conf.items():
        print(conf, end='\n')
        conf['data1'] = dataset[conf['text1_corpus']]
        conf['data2'] = dataset[conf['text2_corpus']]
        generator = inputs.get(conf['input_type'])
        eval_gen[tag] = generator( config = conf )

    for tag, conf in input_eval_loss_conf.items():
        print(conf, end='\n')
        conf['data1'] = dataset[conf['text1_corpus']]
        conf['data2'] = dataset[conf['text2_corpus']]
        generator = inputs.get(conf['input_type'])
        eval_loss_gen[tag] = generator(config=conf)


    ######### Load Model #########
    model = load_model(config)

    loss = []
    for lobj in config['losses']:
        if lobj['object_name'] in mz_specialized_losses:
            loss.append(rank_losses.get(lobj['object_name'])(lobj['object_params']))
        else:
            loss.append(rank_losses.get(lobj['object_name']))
        for k, v in stats_for_plots.items():
            if 'loss' in k:
                stats_for_plots[k][lobj['object_name']] = []
    eval_metrics = OrderedDict()
    for mobj in config['metrics']:
        mobj = mobj.lower()
        if '@' in mobj:
            mt_key, mt_val = mobj.split('@', 1)
            eval_metrics[mobj] = metrics.get(mt_key)(int(mt_val))
        else:
            eval_metrics[mobj] = metrics.get(mobj)
        for k, v in stats_for_plots.items():
            if 'loss' not in k:
                stats_for_plots[k][mobj] = []
    model.compile(optimizer=optimizer, loss=loss)
    print('[Model] Model Compile Done.', end='\n')

    for i_e in range(num_iters):
        for tag, generator in train_gen.items():
            genfun = generator.get_batch_generator()
            evalfun =eval_loss_gen['test_loss'].get_batch_generator()
            print('*' * 100)

            history = model.fit_generator(
                    genfun,
                    steps_per_epoch = display_interval,
                    epochs = 1,
                    shuffle=False,
                    verbose = 0,
                    validation_data=evalfun,
                    validation_steps=display_interval,
                    callbacks=[TrainValTensorBoard(log_dir=os.path.join(logs_dir, 'tensorboard'),
                                                   global_step=display_interval * i_e,
                                                   write_graph=False
                                                   )]
                ) #callbacks=[eval_map])
            for k, v in stats_for_plots.items():
                if 'loss' in k:
                    print('[%s]\t[Train:%s] ' % (time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())), k),
                          end='')
                    if any(srchstr in k for srchstr in ('test','val','valid')):
                        _l = history.history['val_loss'][0]
                        stats_for_plots[k][lobj['object_name']].append(_l)
                    else:
                        _l = history.history['loss'][0]
                        stats_for_plots[k][lobj['object_name']].append(_l)
                    print('Iter:%d\t Loss =%.6f' % (i_e, _l), end='\n')
            print('-' * 50)

        for tag, generator in eval_gen.items():
            genfun = generator.get_batch_generator()
            print('[%s]\t[Eval:%s] ' % (time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())), tag), end='')
            res = dict([[k,0.] for k in eval_metrics.keys()])
            num_valid = 0
            #history_eval = model.evaluate_generator(genfun, steps=1)
            #print("history_eval: {}".format(history_eval))
            for input_data, y_true in genfun:
                y_pred = model.predict(input_data, batch_size=len(y_true))
                if issubclass(type(generator), inputs.list_generator.ListBasicGenerator):
                    list_counts = input_data['list_counts']
                    for k, eval_func in eval_metrics.items():
                        for lc_idx in range(len(list_counts)-1):
                            pre = list_counts[lc_idx]
                            suf = list_counts[lc_idx+1]
                            res[k] += eval_func(y_true = y_true[pre:suf], y_pred = y_pred[pre:suf])
                    num_valid += len(list_counts) - 1
                else:
                    for k, eval_func in eval_metrics.items():
                        res[k] += eval_func(y_true = y_true, y_pred = y_pred)
                    num_valid += 1
            generator.reset()
            for k, v in res.items():
                stats_for_plots[tag][k].append(v/num_valid)

            print('Iter:%d\t%s' % (i_e, '\t'.join(['%s=%f'%(k,v/num_valid) for k, v in res.items()])), end='\n')
            sys.stdout.flush()
        if (i_e+1) % save_weights_iters == 0:
            model.save_weights(weights_file % (i_e+1))
    export_loss(False, 1)
    export_metrics(False, 1)
def exit_handler():
    if len(stats_for_plots) > 0:
        # pass
        export_loss(False, 2)
        export_metrics(False, 2)

def export_loss(is_terminated, verbose):
    legend = []
    data = []
    for k, v in stats_for_plots.items():
        if 'loss' in k:
            for loss in v:
                data.append(v[loss])
                legend.append(k)
    if verbose == 1 or verbose == 3:
        for d in data:
            plt.plot(d)
        plt.title('Train-Test {}'.format(loss))
        plt.ylabel("{}".format(loss))
        plt.xlabel("Iterations over Whole Dataset")
        plt.legend(legend, loc='upper right')
        plt.grid()
        name = os.path.join(logs_dir, 'train_test_loss_plt_{}.png'.format(time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time()))))
        fig = plt.gcf()
        if not is_terminated:
            fig.savefig(name)
            plt.show()
            fig.savefig(name)
            plt.close()
            plt.clf()
        else:
            plt.show()
            plt.close()
            plt.clf()
    elif verbose == 2 or verbose == 3:
        data = np.asarray(data)
        data = np.transpose(data)
        name = os.path.join(logs_dir, 'train_test_loss_{}.csv'.format(
                                                                time.strftime('%m-%d-%Y %H:%M:%S',
                                                                              time.localtime(time.time()))))
        np.savetxt(name, data, delimiter=',', header=','.join(legend))  # X is an array

    print('{} is exported.'.format(name))
def export_metrics(is_terminated, verbose):
    for k, v in stats_for_plots.items():
        if 'loss' not in k:
            for metric in v:
                if verbose == 1 or verbose == 3:
                    plt.plot(v[metric])
                    plt.title('{}: {}'.format(k, metric))
                    plt.ylabel("{}".format(metric))
                    plt.xlabel("Iterations over Whole Dataset")
                    plt.legend(k, loc='upper right')
                    plt.grid()
                    name = os.path.join(logs_dir, '{}_{}_plt_{}.png'.format(k, metric, time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time()))))
                    fig = plt.gcf()
                    if not is_terminated:
                        fig.savefig(name)
                        plt.show()
                        fig.savefig(name)
                        plt.close()
                        plt.clf()
                        print('{} is exported.'.format(name))
                    else:
                        plt.show()
                        # plt.close()
                        plt.clf()
                elif verbose == 2 or verbose == 3:

                    name = os.path.join(logs_dir, '{}_{}_{}.csv'.format(k, metric,
                                                                            time.strftime('%m-%d-%Y %H:%M:%S',
                                                                                          time.localtime(time.time()))))
                    np.savetxt(name, v[metric], delimiter=',', header=metric)  # X is an array
                print('{} is exported.'.format(name))

atexit.register(exit_handler)

def predict(config):
    ######## Read input config ########

    print(json.dumps(config, indent=2), end='\n')
    input_conf = config['inputs']
    share_input_conf = input_conf['share']

    # collect embedding
    if 'embed_path' in share_input_conf:
        embed_dict = read_embedding(filename=share_input_conf['embed_path'])
        _PAD_ = share_input_conf['vocab_size'] - 1
        embed_dict[_PAD_] = np.zeros((share_input_conf['embed_size'], ), dtype=np.float32)
        embed = np.float32(np.random.uniform(-0.02, 0.02, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = convert_embed_2_numpy(embed_dict, embed = embed)
    else:
        embed = np.float32(np.random.uniform(-0.2, 0.2, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = embed
    print('[Embedding] Embedding Load Done.', end='\n')

    # list all input tags and construct tags config
    input_predict_conf = OrderedDict()
    for tag in input_conf.keys():
        if 'phase' not in input_conf[tag]:
            continue
        if input_conf[tag]['phase'] == 'PREDICT':
            input_predict_conf[tag] = {}
            input_predict_conf[tag].update(share_input_conf)
            input_predict_conf[tag].update(input_conf[tag])
    print('[Input] Process Input Tags. %s in PREDICT.' % (input_predict_conf.keys()), end='\n')

    # collect dataset identification
    dataset = {}
    for tag in input_conf:
        if tag == 'share' or input_conf[tag]['phase'] == 'PREDICT':
            if 'text1_corpus' in input_conf[tag]:
                datapath = input_conf[tag]['text1_corpus']
                if datapath not in dataset:
                    dataset[datapath], _ = read_data(datapath)
            if 'text2_corpus' in input_conf[tag]:
                datapath = input_conf[tag]['text2_corpus']
                if datapath not in dataset:
                    dataset[datapath], _ = read_data(datapath)
    print('[Dataset] %s Dataset Load Done.' % len(dataset), end='\n')

    # initial data generator
    predict_gen = OrderedDict()

    for tag, conf in input_predict_conf.items():
        print(conf, end='\n')
        conf['data1'] = dataset[conf['text1_corpus']]
        conf['data2'] = dataset[conf['text2_corpus']]
        generator = inputs.get(conf['input_type'])
        predict_gen[tag] = generator(
                                    #data1 = dataset[conf['text1_corpus']],
                                    #data2 = dataset[conf['text2_corpus']],
                                     config = conf )

    ######## Read output config ########
    output_conf = config['outputs']

    ######## Load Model ########
    global_conf = config["global"]
    weights_file = str(global_conf['weights_file']) + '.' + str(global_conf['test_weights_iters'])

    model = load_model(config)
    model.load_weights(weights_file)

    eval_metrics = OrderedDict()
    for mobj in config['metrics']:
        mobj = mobj.lower()
        if '@' in mobj:
            mt_key, mt_val = mobj.split('@', 1)
            eval_metrics[mobj] = metrics.get(mt_key)(int(mt_val))
        else:
            eval_metrics[mobj] = metrics.get(mobj)
    res = dict([[k,0.] for k in eval_metrics.keys()])

    for tag, generator in predict_gen.items():
        genfun = generator.get_batch_generator()
        print('[%s]\t[Predict] @ %s ' % (time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())), tag), end='')
        num_valid = 0
        res_scores = {}
        for input_data, y_true in genfun:
            y_pred = model.predict(input_data, batch_size=len(y_true) )

            if issubclass(type(generator), inputs.list_generator.ListBasicGenerator):
                list_counts = input_data['list_counts']
                for k, eval_func in eval_metrics.items():
                    for lc_idx in range(len(list_counts)-1):
                        pre = list_counts[lc_idx]
                        suf = list_counts[lc_idx+1]
                        res[k] += eval_func(y_true = y_true[pre:suf], y_pred = y_pred[pre:suf])

                y_pred = np.squeeze(y_pred)
                for lc_idx in range(len(list_counts)-1):
                    pre = list_counts[lc_idx]
                    suf = list_counts[lc_idx+1]
                    for p, y, t in zip(input_data['ID'][pre:suf], y_pred[pre:suf], y_true[pre:suf]):
                        if p[0] not in res_scores:
                            res_scores[p[0]] = {}
                        res_scores[p[0]][p[1]] = (y, t)

                num_valid += len(list_counts) - 1
            else:
                for k, eval_func in eval_metrics.items():
                    res[k] += eval_func(y_true = y_true, y_pred = y_pred)
                for p, y, t in zip(input_data['ID'], y_pred, y_true):
                    if p[0] not in res_scores:
                        res_scores[p[0]] = {}
                    res_scores[p[0]][p[1]] = (y[1], t[1])
                num_valid += 1
        generator.reset()

        if tag in output_conf:
            if output_conf[tag]['save_format'] == 'TREC':
                with open(output_conf[tag]['save_path'], 'w') as f:
                    for qid, dinfo in res_scores.items():
                        dinfo = sorted(dinfo.items(), key=lambda d:d[1][0], reverse=True)
                        for inum,(did, (score, gt)) in enumerate(dinfo):
                            f.write('%s\tQ0\t%s\t%d\t%f\t%s\t%s\n'%(qid, did, inum, score, config['net_name'], gt))
            elif output_conf[tag]['save_format'] == 'TEXTNET':
                with open(output_conf[tag]['save_path'], 'w') as f:
                    for qid, dinfo in res_scores.items():
                        dinfo = sorted(dinfo.items(), key=lambda d:d[1][0], reverse=True)
                        for inum,(did, (score, gt)) in enumerate(dinfo):
                            f.write('%s %s %s %s\n'%(gt, qid, did, score))

        print('[Predict] results: ', '\t'.join(['%s=%f'%(k,v/num_valid) for k, v in res.items()]), end='\n')
        sys.stdout.flush()

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', help='Phase: Can be train or predict, the default value is train.')
    parser.add_argument('--model_file', default='./models/arci.config', help='Model_file: MatchZoo model file for the chosen model.')
    args = parser.parse_args()
    model_file =  args.model_file
    with open(model_file, 'r') as f:
        config = json.load(f)
    phase = args.phase
    if args.phase == 'train':
        train(config)
    elif args.phase == 'predict':
        predict(config)
    else:
        print('Phase Error.', end='\n')
    return

if __name__=='__main__':
    main(sys.argv)
