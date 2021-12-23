import json
import os
import pickle
import re
import traceback
from typing import List
import random

import pandas as pd
import argparse

from stopwords_tfidf_generator import do_generate_stopwords
from tfidf_classifier import do_classify
from utils import Question, load_data_kse, postag_filter
from vncorenlp import VnCoreNLP

tokenizer_obj = VnCoreNLP("vncorenlp_data/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 
 
def vi_tokenize(str_in):
    # return str_in
    str_in = " ".join(str_in.split(" ")[:512])
    sentences = [" ".join(s) for s in tokenizer_obj.tokenize(str_in)]
  
    return " ".join(sentences)
 
def generate_pair_inputs(data_pred, data_gold, _c_keys, append_gold=False, sub_doc_info=None):
    _data_pairs_id = []
    _, _, sub_key_mapping = sub_doc_info or (None, None, {})

    for i in range(data_pred.shape[0]):
        cur_pred = [_c_keys[idx] for idx in data_pred[i]]
        cur_label = [a.get_id() for a in data_gold[i].relevant_a]
        q_id = data_gold[i].id

        if append_gold:
            for id_civil_lb in cur_label:
                if id_civil_lb not in cur_pred:
                    cur_pred = cur_pred + [id_civil_lb]

        for j, id_civil_pred in enumerate(cur_pred):
            check_lb = id_civil_pred in cur_label
            _data_pairs_id.append(((id_civil_pred, q_id), check_lb))

            # append sub articles (by chunking)
            for id_c in sub_key_mapping.get(id_civil_pred, []):
                _data_pairs_id.append(((id_c, q_id), check_lb))

    print('Number data pairs: ', len(_data_pairs_id))
    return _data_pairs_id
 

def aggregate_sentence_pairs(_c_docs, _c_keys, _data_pairs_id, _q: List[Question], plus_filter_postags=False, filter_lb=False,
                             empty_article_id="None", sub_doc_info=None):
    _new_dataset = []
    _q_map = dict((q.id, q.content) for q in _q)
    empty_article_content = ""
    _c_docs = _c_docs + [empty_article_content]
    _c_keys = _c_keys + [empty_article_id]

    _c_sub_docs, _c_sub_keys, _ = sub_doc_info or (None, None, {})
    _c_docs = _c_docs + (_c_sub_docs if _c_sub_docs is not None else [])
    _c_keys = _c_keys + (_c_sub_keys if _c_sub_keys is not None else [])

    for (id_civil_pred, q_id), lb in _data_pairs_id:
        try:
            _new_dataset.append({
                "id": [id_civil_pred, q_id],
                "c_code": _c_docs[_c_keys.index(id_civil_pred)],
                "query": _q_map[q_id],
                'label': lb
            })

            if plus_filter_postags:
                if filter_lb and lb:
                    _new_dataset.append({
                        "id": [id_civil_pred + "_pos_filtered", q_id],
                        "c_code": postag_filter(_c_docs[_c_keys.index(id_civil_pred)]),
                        "query": _q_map[q_id],
                        'label': lb
                    })
                if not filter_lb:
                    _new_dataset.append({
                        "id": [id_civil_pred + "_pos_filtered", q_id],
                        "c_code": postag_filter(_c_docs[_c_keys.index(id_civil_pred)]),
                        "query": _q_map[q_id],
                        'label': lb
                    })
        except Exception as e:
            traceback.print_stack()
            print(e)
            print("[Err] {}".format(((id_civil_pred, q_id), lb)))
    return _new_dataset


def aggregate_sentence_pairs_task5(_data_pairs_id, _q: List[Question]):
    _new_dataset = []
    _q_map = dict((q.id, q.content) for q in _q)

    for q_id, lb in _data_pairs_id:
        _new_dataset.append({
            "id": q_id,
            "query": _q_map[q_id],
            'label': lb
        })
    return _new_dataset


def gen_mrpc_data(coliee_data_, file_path):
    data = {
        "label": [],
        "#1 ID": [],
        "#2 ID": [],
        "sentence1": [],
        "sentence2": [],
    }
    for e in coliee_data_:
        data['label'].append(1 if e['label'] else 0)
        data['#1 ID'].append(e['id'][1])
        data['#2 ID'].append(e['id'][0])
        data['sentence1'].append(e['query'].replace('\n', " "))
        data['sentence2'].append(e['c_code'].replace('\n', " "))
    df = pd.DataFrame(data=data)
    df.to_csv(file_path, index=False, sep=',')


def gen_cola_data(coliee_data_, file_path):
    data = {
        "sentence": [],
        "label": [],
        "id": [],
    }
    for e in coliee_data_:
        data['label'].append(1 if e['label'] else 0)
        data['sentence'].append(e['query'].replace('\n', " "))
        data['id'].append(e['id'].replace('\n', " "))
    df = pd.DataFrame(data=data)
    df.to_csv(file_path, index=False, sep=',')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_folder_base',
                        action="store", dest="path_folder_base",
                        help="path folder saving data", default='path/to/path_folder_base')
    parser.add_argument('--path_output_dir',
                        action="store", dest="path_output_dir",
                        help="path folder saving output data", default='path/to/path_output_dir')
    parser.add_argument('--type_data',
                        action="store", dest="type_data",
                        help="type data for generating process: task3 | task4", default='task3')
    parser.add_argument('--test_file',
                        action="store", dest="test_file", type=str,
                        help="path to the test file", default=None)
    parser.add_argument('--topk',
                        action="store", dest="topk", type=int,
                        help="topk select by tfidf when generating data", default=150) 
    parser.add_argument('--only_test',
                        action="store_true", dest="only_test", 
                        help="just generate testing data", default=False)
    parser.add_argument('--chunk_content_size',
                        action="store", dest="chunk_content_size", type=int,
                        help="chunk content of article with size", default=0)
    parser.add_argument('--chunk_content_stride',
                        action="store", dest="chunk_content_stride", type=int,
                        help="chunk content of article with stride", default=0)
    parser.add_argument('--tok',
                    action="store_true", dest="tok",  
                    help="run tokenize", default=False) 

    options = parser.parse_args()
    tokenizer = vi_tokenize if options.tok else None

    path_folder_base = options.path_folder_base
    topk_select = options.topk

    chunk_content_info = [options.chunk_content_size,
                          options.chunk_content_stride] \
        if options.chunk_content_size > 0 and options.chunk_content_stride > 0 else None
        
    test_ids = None
    if options.test_file is not None:
        test_data = json.load(open("{}/{}".format(path_folder_base, options.test_file)))
        if 'items' in test_data:
            test_data = test_data['items']
        test_ids = [s["question_id"] for s in test_data]
    
    path_data_cached = '{}/tokenized_data_cached.pkl'.format(options.path_output_dir)
    if os.path.isfile(path_data_cached):
        print ("Load cached file data: {}".format(path_data_cached))
        c_docs, c_keys, dev_q, test_q, train_q, sub_doc_info = pickle.load(open(path_data_cached, 'rb'))
    else:
        print ("Load data and tokenize data")
        c_docs, c_keys, dev_q, test_q, train_q, sub_doc_info = load_data_kse(
            path_folder_base=path_folder_base,  ids_test=test_ids, tokenizer=tokenizer, testing_data=options.test_file, 
            chunk_content_info=chunk_content_info
            )
        try:
            pickle.dump((c_docs, c_keys, dev_q, test_q, train_q, sub_doc_info), open(path_data_cached, 'wb'))
        except Exception as e:
            print(e)
        
    c_docs_raw = sub_doc_info[3]
    sub_doc_info = sub_doc_info[:3] 

    # test_q = train_q
    if len(dev_q) == 0:
        dev_q = train_q
    if len(test_q) == 0:
        test_q = train_q
 
    stopwords = None

    # build tfidf vectorizer and generate pair sentence for training process
    # if text is tokenized, not combine tfidf with bm25, otherwise combine
    if not options.only_test:
        train_pred, (_, _, _, vectorizer) = do_classify(c_docs, c_keys, train_q,
                                                        stopwords_=stopwords, topk=topk_select, tokenizer=tokenizer, combine_score=(not options.tok))
        train_data_pairs_id = generate_pair_inputs(_c_keys=c_keys, data_pred=train_pred, data_gold=train_q,
                                                append_gold=True, sub_doc_info=sub_doc_info)
    else:
        train_data_pairs_id = []

    test_pred, _ = do_classify(
        c_docs, c_keys, test_q, vectorizer=vectorizer, topk=topk_select, tokenizer=tokenizer, combine_score=(not options.tok))
    test_data_pairs_id = generate_pair_inputs(
        _c_keys=c_keys, data_pred=test_pred, data_gold=test_q, sub_doc_info=sub_doc_info)

    dev_pred, _ = do_classify(
        c_docs, c_keys, dev_q, vectorizer=vectorizer, topk=topk_select, tokenizer=tokenizer, combine_score=(not options.tok))
    dev_data_pairs_id = generate_pair_inputs(
        _c_keys=c_keys, data_pred=dev_pred, data_gold=dev_q, sub_doc_info=sub_doc_info)  

    print("len(train_data_pairs_id), len(test_data_pairs_id), len(dev_data_pairs_id) = ",
          len(train_data_pairs_id), len(test_data_pairs_id), len(dev_data_pairs_id))

    # save file csv following template of mrpc task
    path_folder_data_out = options.path_output_dir
    if not os.path.exists(path_folder_data_out):
        os.mkdir(path_folder_data_out)

    # fill data from train/test data_pairs_id
    new_dataset_train = aggregate_sentence_pairs(c_docs, c_keys, train_data_pairs_id, train_q,
                                                plus_filter_postags=False,
                                                filter_lb=False, sub_doc_info=sub_doc_info)
    new_dataset_test = aggregate_sentence_pairs(c_docs, c_keys, test_data_pairs_id, test_q,
                                                plus_filter_postags=False,
                                                filter_lb=False, sub_doc_info=sub_doc_info)
    new_dataset_dev = aggregate_sentence_pairs(c_docs, c_keys, dev_data_pairs_id, dev_q,
                                            plus_filter_postags=False,
                                            filter_lb=False, sub_doc_info=sub_doc_info)

    gen_mrpc_data(new_dataset_train,
                "{}/train.csv".format(path_folder_data_out))
    gen_mrpc_data(new_dataset_test, "{}/test.csv".format(path_folder_data_out))
    gen_mrpc_data(new_dataset_dev, "{}/dev.csv".format(path_folder_data_out))

    # save tfidf vectorizer that filter fop 150 civil document
    pickle.dump(vectorizer, open(
        "{}/tfidf_classifier.pkl".format(path_folder_data_out), "wb")) 


