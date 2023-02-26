import json
import os
import pickle
import re
import traceback

import fugashi
import pandas as pd
import argparse

from data_utils.stopwords_tfidf_generator import do_generate_stopwords
from data_utils.tfidf_classifier import do_classify
from data_utils.utils import load_data_coliee, postag_filter

# The Tagger object holds state about the dictionary.
jp_tagger = fugashi.Tagger()


def jp_tokenize(text):
    return [word.surface for word in jp_tagger(text)]


def generate_pair_inputs(data_pred, data_gold, _c_keys, append_gold=False, sub_doc_info=None):
    _data_pairs_id = []
    _, _, sub_key_mapping = sub_doc_info or (None, None, {})

    for i in range(data_pred.shape[0]):
        cur_pred = [_c_keys[idx] for idx in data_pred[i]]
        cur_label = data_gold[i]["result"]
        q_id = data_gold[i]['index']

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


def generate_pair_inputs_task4(data_pred, data_gold, _c_keys, testing=False, bert_predictions=None, sub_doc_info=None):
    _data_pairs_id = []
    _, _, sub_key_mapping = sub_doc_info or (None, None, {})
    bert_predictions = bert_predictions or {}
    for i in range(data_pred.shape[0]):
        relevant_article_ids = data_gold[i]["result"]
        q_id = data_gold[i]['index']
        label = True if data_gold[i]['label'] == 'Y' else False

        # get predictions from bert + tfidf
        cur_pred = list(set([e[1] for e in bert_predictions.get(q_id, [])] +
                            [_c_keys[idx] for idx in data_pred[i]]))

        for id_civil_lb in relevant_article_ids:
            if id_civil_lb not in cur_pred:
                cur_pred = cur_pred + [id_civil_lb]

        if not testing:
            # _data_pairs_id.append(
            #     (("None", q_id), label))
            for j, id_civil_pred in enumerate(cur_pred):
                if id_civil_pred in relevant_article_ids:
                    _data_pairs_id.append(
                        ((id_civil_pred, q_id), label))
                else:
                    _data_pairs_id.append(((id_civil_pred, q_id), False))
        else:
            for id_article in relevant_article_ids:
                _data_pairs_id.append(
                    ((id_article, q_id), label))
        
    if sub_key_mapping is not None:
        for e in _data_pairs_id:
            for sub_art in sub_key_mapping.get(e[0][0], []):
                _data_pairs_id.append(((sub_art, e[0][1]), e[1]))

    print(len(_data_pairs_id))
    return _data_pairs_id


def aggregate_sentence_pairs(_c_docs, _c_keys, _data_pairs_id, _q, plus_filter_postags=False, filter_lb=False,
                             empty_article_id="None", sub_doc_info=None):
    _new_dataset = []
    _q_map = dict((q["index"], q['content']) for q in _q)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_folder_base',
                        action="store", dest="path_folder_base",
                        help="path folder saving data", default='path/to/path_folder_base')
    parser.add_argument('--meta_data_alignment',
                        action="store", dest="meta_data_alignment",
                        help="path to folder alignment with folder data base, should be in english", default='path/to/meta_data_alignment')
    parser.add_argument('--path_output_dir',
                        action="store", dest="path_output_dir",
                        help="path folder saving output data", default='path/to/path_output_dir')
    parser.add_argument('--lang',
                        action="store", dest="lang",
                        help="language: en | jp", default='en')
    parser.add_argument('--type_data',
                        action="store", dest="type_data",
                        help="type data for generating process: task3 | task4", default='task3')
    parser.add_argument('--dev_ids',
                        action="extend", dest="dev_ids", type=str, nargs="*",
                        help="id for dev set", default=[])
    parser.add_argument('--test_ids',
                        action="extend", dest="test_ids", type=str, nargs="*",
                        help="id for test set", default=[])
    parser.add_argument('--test_file',
                        action="store", dest="test_file", type=str,  
                        help="path to the test file", default=None)
    parser.add_argument('--chunk_content_size',
                        action="store", dest="chunk_content_size", type=int,
                        help="chunk content of article with size", default=0)
    parser.add_argument('--chunk_content_stride',
                        action="store", dest="chunk_content_stride", type=int,
                        help="chunk content of article with stride", default=0)
    parser.add_argument('--topk',
                        action="store", dest="topk", type=int,
                        help="topk select by tfidf when generating data", default=150)
    parser.add_argument('--faked_result',
                        action="store", dest="faked_result", type=str,
                        help="topk select by tfidf when generating data", default="")
    options = parser.parse_args()

    path_folder_base = options.path_folder_base
    if options.lang == 'en':
        options.meta_data_alignment = path_folder_base
    meta_data_alignment = options.meta_data_alignment
    lang = options.lang
    topk_select = options.topk
    postags_select = ["V", "N", "P", "."]
    if lang == 'jp':
        tokenizer = jp_tokenize
    else:
        tokenizer = None

    chunk_content_info = [options.chunk_content_size,
                          options.chunk_content_stride] \
        if options.chunk_content_size > 0 and options.chunk_content_stride > 0 else None
    c_docs, c_keys, dev_q, test_q, train_q, sub_doc_info = load_data_coliee(path_folder_base=path_folder_base,
                                                                            postags_select=None,
                                                                            ids_test=options.test_ids,
                                                                            ids_dev=options.dev_ids,
                                                                            lang=lang,
                                                                            path_data_alignment=meta_data_alignment,
                                                                            chunk_content_info=chunk_content_info,
                                                                            tokenizer=tokenizer,
                                                                            test_file=options.test_file)
    
    # save file csv following template of mrpc task
    path_folder_data_out = options.path_output_dir
    if not os.path.exists(path_folder_data_out):
        os.mkdir(path_folder_data_out)
        
    json.dump({
        'c_docs': c_docs, 
        'c_keys': c_keys, 
        'dev_q': dev_q, 
        'test_q':test_q, 
        'train_q':train_q, 
        'sub_doc_info': sub_doc_info
        }, open(f'{options.path_output_dir}/all_data.json', 'wt'), indent=2, ensure_ascii=False 
    )
    # test_q = train_q
    if len(dev_q) == 0:
        dev_q = train_q
    if len(test_q) == 0:
        test_q = train_q
        

    # build japanese tokenizer

    # load stopwords generated before
    do_generate_stopwords(path_folder_base, threshold=0.00, tokenizer=tokenizer, data=(
        c_docs, c_keys, dev_q, test_q, train_q, sub_doc_info))  # code to generate stop words automatic using tfidf
    stopwords = json.load(
        open("{}/stopwords/stopwords.json".format(path_folder_base), "rt"))

    # build tfidf vectorizer and generate pair sentence for training process
    train_pred, (_, _, _, vectorizer) = do_classify(c_docs, c_keys, train_q,
                                                    stopwords_=stopwords, topk=topk_select, tokenizer=tokenizer)
                                                    
    if options.type_data == 'task3':
        train_data_pairs_id = generate_pair_inputs(_c_keys=c_keys, data_pred=train_pred, data_gold=train_q,
                                                   append_gold=True, sub_doc_info=sub_doc_info)
    else:
        train_data_pairs_id = generate_pair_inputs_task4(_c_keys=c_keys, data_pred=train_pred, data_gold=train_q,
                                                         testing=False, sub_doc_info=sub_doc_info)

    test_pred, _ = do_classify(
        c_docs, c_keys, test_q, vectorizer=vectorizer, topk=topk_select, tokenizer=tokenizer)
    if options.type_data == 'task3':
        test_data_pairs_id = generate_pair_inputs(
            _c_keys=c_keys, data_pred=test_pred, data_gold=test_q, sub_doc_info=sub_doc_info)
    else:
        test_data_pairs_id = generate_pair_inputs_task4(_c_keys=c_keys, data_pred=test_pred, data_gold=test_q,
                                                        testing=True, sub_doc_info=sub_doc_info)

    dev_pred, _ = do_classify(
        c_docs, c_keys, dev_q, vectorizer=vectorizer, topk=topk_select, tokenizer=tokenizer)
    if options.type_data == 'task3':
        dev_data_pairs_id = generate_pair_inputs(
            _c_keys=c_keys, data_pred=dev_pred, data_gold=dev_q, sub_doc_info=sub_doc_info)
    else:
        dev_data_pairs_id = generate_pair_inputs_task4(_c_keys=c_keys, data_pred=dev_pred, data_gold=dev_q,
                                                       testing=True, sub_doc_info=sub_doc_info)

    print("len(train_data_pairs_id), len(test_data_pairs_id), len(dev_data_pairs_id) = ",
          len(train_data_pairs_id), len(test_data_pairs_id), len(dev_data_pairs_id))

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


    # run fake data (silver data) if found faked result 
    faked_result_modified_log = []
    if len(options.faked_result) > 0 and chunk_content_info is not None:
        faked_result = pd.read_csv(open(options.faked_result, "rt"), sep='\t')
        if len(faked_result) == len(new_dataset_train):
            for i_line in range(len(faked_result)):
                if 'sub' in new_dataset_train[i_line]['id'][0] and \
                        new_dataset_train[i_line]['label'] and faked_result['prediction'][i_line] == 0:
                    new_dataset_train[i_line]['label'] = False
                    faked_result_modified_log.append(new_dataset_train[i_line])
        
            gen_mrpc_data(faked_result_modified_log,
                        "{}/log_train_modified.csv".format(path_folder_data_out))
            print("Len faked_result_modified_log = {}".format(len(faked_result_modified_log)))

    gen_mrpc_data(new_dataset_train,
                  "{}/train.csv".format(path_folder_data_out))
    gen_mrpc_data(new_dataset_test, "{}/test.csv".format(path_folder_data_out))
    gen_mrpc_data(new_dataset_dev, "{}/dev.csv".format(path_folder_data_out))

    # save tfidf vectorizer that filter fop 150 civil document
    pickle.dump(vectorizer, open(
        "{}/tfidf_classifier.pkl".format(path_folder_data_out), "wb"))
