from pprint import pprint
from typing import Any, Dict, List

import nltk
import glob
import pickle
import torch
import pandas as pd
import re
import json
import numpy as np

rdrsegmenter = None


def standardize_data(row):
    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = re.sub(r"[\.,\?]+$-", "", row)
    # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
    row = row.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("-", " ").replace("?", " ") \
        .replace("(", " ").replace(")", " ")
    row = re.sub(r" {2, }", " ", row).strip().lower()
    return row

 
class Article:
    pattern = "{}-->{}-->{}"

    def __init__(self, a_id, l_id, content, content_raw=None) -> None:
        self.a_id = a_id
        self.l_id = l_id
        self.content = content
        self.content_raw = content_raw if content_raw is not None else content

    def __str__(self) -> str:
        return self.pattern.format(self.l_id, self.a_id, self.content)

    @classmethod
    def from_string(cls, str_in):
        info = str_in.split(cls.pattern.format("", "a", "").split("a")[0])
        return cls(info[1], info[0], info[2])

    def get_id(self):
        return self.pattern.format(self.l_id, self.a_id, "")

    def get_subid(self, sub_id):
        return self.pattern.format(self.l_id, self.a_id+"-sub{}".format(sub_id), "")


class Question:
    pattern = "{}-->{}"

    def __init__(self, id, content, relevant_a: List[Article] = None, label: bool = None, content_raw=None) -> None:
        self.id = id
        self.content = content
        self.relevant_a = relevant_a or []
        self.label = label
        self.content_raw = content_raw if content_raw is not None else content

    def __str__(self) -> str:
        return self.pattern.format(self.id, self.content)

    @classmethod
    def from_string(cls, str_in):
        info = str_in.split(cls.pattern.format("", "a", "").split("a")[0])
        return cls(info[0], info[1])

    def get_id(self):
        return self.pattern.format(self.id, "")


def f_score(p, r, beta=1):
    y = (beta * beta * p + r)
    return (1 + beta * beta) * p * r / y if y != 0 else 0.0


def micro_result(count_real_lb, count_predicted, count_true,  count_gold_lb=131):
    p = count_true/count_predicted if count_predicted != 0 else 0.0
    r = count_true/count_real_lb if count_real_lb != 0 else 0.0
    result = {"count_real_lb": count_real_lb,
              "count_predicted": count_predicted,
              "count_gold_lb": count_gold_lb,
              "count_true": count_true,
              "P": p,
              "R": r,
              "f1": f_score(p, r, 1),
              "f2": f_score(p, r, 2),
              "f2_": f_score(p, count_true/count_gold_lb, 2)}
    print(result)
    return result


def evaluate_by_similarity(similarities_, gold_data, c_keys, topk=150):
    count_true = 0
    count_all_prediction = 0
    count_all_gold_lb = 0

    idx_result = similarities_.argsort()[:, -topk:]
    for i in range(idx_result.shape[0]):
        gold_lb = gold_data[i]['result']
        count_all_gold_lb += len(gold_lb)

        pred = [c_keys[idx] for idx in idx_result[i]]
        count_all_prediction += len(pred)

        for i, pred_lb in enumerate(pred):
            if pred_lb in gold_lb:
                count_true += 1

    print(count_true, count_all_prediction,  count_all_gold_lb,
          'P: ', count_true/count_all_prediction,
          'R: ', count_true/count_all_gold_lb,
          'F1: ', f_score(count_true*1.0/count_all_prediction,
                          count_true*1.0/count_all_gold_lb),
          'F2: ', f_score(count_true*1.0/count_all_prediction,
                          count_true*1.0/count_all_gold_lb, beta=2),
          )
    return idx_result


def evaluate_by_label(prediction_file, test_dat_file, ensemble_files=None):
    test_dat = pd.read_csv(test_dat_file, sep=',')
    predictions = []

    count_real_lb = 0
    count_gold_lb = 131
    count_true = 0
    count_predicted = 0
    ensemble_files = ensemble_files or []

    if prediction_file not in ensemble_files:
        ensemble_files.append(prediction_file)

    for pred_file in ensemble_files:
        prediction_ = pd.read_csv(pred_file, sep='\t')
        predictions.append(prediction_)

    for i in range(len(test_dat)):
        if test_dat['label'][i] == 1:
            count_real_lb += 1
            for prediction in predictions:
                if prediction['prediction'][i] == 1:
                    count_true += 1
                    break

        for prediction in predictions:
            if prediction['prediction'][i] == 1:
                count_predicted += 1
                break

    return micro_result(count_real_lb, count_predicted, count_true,  count_gold_lb)

def evaluate_idx(preds, gold_data, c_keys=None):
    try:
        count_true = 0
        count_all_prediction = 0
        count_all_gold_lb = 0

        for i_gold in range(len(preds)):
            gold_lb = [a.get_id() for a in gold_data[i_gold].relevant_a]
            count_all_gold_lb += len(gold_lb)

            pred = [c_keys[idx] for idx in preds[i_gold]]
            count_all_prediction += len(pred)

            for _i, pred_lb in enumerate(pred):
                if pred_lb in gold_lb:
                    count_true += 1

        print(count_true, count_all_prediction, count_all_gold_lb,
              'P: ', count_true / count_all_prediction,
              'R: ', count_true / count_all_gold_lb,
              'F1: ', f_score(count_true * 1.0 / count_all_prediction,
                              count_true * 1.0 / count_all_gold_lb),
              'F2: ', f_score(count_true * 1.0 / count_all_prediction,
                              count_true * 1.0 / count_all_gold_lb, beta=2),
              )
        return preds
    except Exception as e:
        print(e)
    return preds

def combine_idxs(idx_ifidf, idx_bm25, top_k=100):
    preds = []
    for i in range(len(idx_bm25)):
        prediction_item = []
        for j in range(len(idx_bm25[i])):
            if idx_bm25[i][j] not in prediction_item:
                prediction_item.append(idx_bm25[i][j])

            if len(prediction_item) == top_k:
                break

            if idx_ifidf[i][j] not in prediction_item:
                prediction_item.append(idx_ifidf[i][j])

            if len(prediction_item) == top_k:
                break
        preds.append(prediction_item)
    return np.array(preds)

def evaluate(similarities_, gold_data, topk=150, c_keys=None):
    try:
        idx_result = similarities_.argsort()[:, -topk:]
        return evaluate_idx(idx_result, gold_data, c_keys)
    except Exception as e:
        print(e)
        return idx_result

def _article_content(full_content, chunk_content_info=None):
    if chunk_content_info is None:
        return ["{}".format(full_content)]
    chunk_content_size, chunk_content_stride = chunk_content_info[0], chunk_content_info[1]
    sub_contents = []
    separate_w = ' ' 
    words = full_content.split(separate_w)

    if len(words) > chunk_content_size:
        for i_start in range(0, len(words), chunk_content_size-chunk_content_stride):
            sub_cont = separate_w.join(
                words[i_start:i_start + chunk_content_size])
            sub_contents.append(sub_cont)
            if len(words[i_start:i_start + chunk_content_size]) < chunk_content_size:
                break

    articles = ["{}".format(full_content)] + ["{}".format(sub_content) for sub_content in sub_contents]
    return articles

     
def _do_nothing(str_in):
    return str_in

def load_data_kse(path_folder_base="data/", postags_select=None, ids_test=None, ids_dev=None, tokenizer=None,
                  law_corpus='legal_corpus.json', training_data='train_question_answer.json', testing_data=None,
                  chunk_content_info=None):

    if tokenizer is None:
        tokenizer = _do_nothing

    articles = {}
    sub_articles = {}
    sub_key_mapping ={}
    articles_raw = json.load(open("{}/{}".format(path_folder_base, law_corpus)))
    for l_info in articles_raw:
        l_id = l_info['id'] if 'id' in l_info else l_info['law_id']
        for a_info in l_info["articles"]:
            a_id = a_info['id'] if 'id' in a_info else a_info['article_id']
            a_title = tokenizer(a_info['title']) + " . " if 'title' in a_info else ""
            a_title_raw =  a_info['title'] + " . " if 'title' in a_info else ""

            a_content_s = _article_content(tokenizer(a_info['text']), chunk_content_info)
            a_content_s_raw = _article_content(a_info['text'], chunk_content_info)

            new_a = Article(l_id=l_id, a_id=a_id, content=a_title + a_content_s[0], content_raw=a_title_raw+a_content_s_raw[0])
            k = new_a.get_id()
            articles[k] = new_a

            for i, a_content in enumerate(a_content_s[1:10]):
                new_sub_a = Article(l_id=l_id, a_id=a_id, content=a_title + a_content)
                sub_articles[new_a.get_subid(i)] = new_sub_a
                
                if k not in sub_key_mapping:
                    sub_key_mapping[k] = []
                sub_key_mapping[k].append(new_sub_a.get_subid(i))

    print(len(articles))
    print(articles[list(articles.keys())[0]])

    # load annotated data
    data = []
    q_raw = json.load(open("{}/{}".format(path_folder_base, training_data)))
    if testing_data is not None:
        q_raw_test = json.load(open("{}/{}".format(path_folder_base, testing_data)))
        if 'items' in q_raw and 'items' in q_raw_test:
            for e in q_raw_test['items']:
                e['relevant_articles'] = e.get('relevant_articles', [])
            q_raw['items'] = q_raw['items'] + q_raw_test['items']
        
    if 'items' in q_raw:
        for q_info in q_raw['items']:
            data.append(Question(id=q_info["question_id"], content=tokenizer(q_info["question"])if tokenizer is not None else q_info["question"],
                content_raw=q_info["question"],
                relevant_a=[articles[Article(
                a_info["article_id"], a_info["law_id"], None).get_id()] for a_info in q_info["relevant_articles"]], 
                label=True))
    else:
        for q_info in q_raw:
            data.append(Question(id=q_info["question_id"], content=q_info["text"], relevant_a=[articles[Article(
                a_info["article_id"], a_info["law_id"], None).get_id()] for a_info in q_info["relevant_articles"]], 
                label=q_info['label']))

    # random test id 
    if ids_test is None:
        ids_test = [q.id for idx, q in enumerate(data) if idx % 10 < 2] 
    if ids_dev is None or len(ids_dev) == 0:
        ids_dev = ids_test
    print('Test ids = {}, Dev ids = {}', ids_test, ids_dev)

    print("Test ids ({} samples) = {}".format(len(ids_test), ids_test))
    test_q = [q for q in data if q.id in ids_test]
    print('Len test_q', len(test_q))
    dev_q = [q for q in data if q.id in ids_dev]
    print('Len dev_q', len(dev_q))
    train_q = [q for q in data if q.id not in set(ids_test + ids_dev)]
    print('Len train_q', len(train_q))

    c_docs = []
    c_docs_raw = []
    c_keys = []
    for k, c in articles.items():
        c_docs.append(c.content)
        c_docs_raw.append(c.content_raw)
        c_keys.append(k)

    c_sub_docs, c_sub_keys = [], []
    for k, c in sub_articles.items():
        c_sub_docs.append(c.content)
        c_sub_keys.append(k)
        
    return c_docs, c_keys, dev_q, test_q, train_q, (c_sub_docs, c_sub_keys, sub_key_mapping, c_docs_raw)


def postag_filter(input_str, tags_filter=["V", "N", "P", "."]):
    words = nltk.word_tokenize(input_str)
    pos = nltk.pos_tag(words)
    new_words = []

    for p in pos:
        if p[1][0] in tags_filter:
            new_words.append(p[0])
    return " ".join(new_words)


def aggregate_results(base_folder, aggregate_predictions=None, keys=None):
    prediction_mt = pickle.load(
        open("{}/predictions.pkl".format(base_folder), "rb"))
    test_dat = pd.read_csv("{}/test.tsv".format(base_folder), sep="\t")
    prediction = pd.read_csv(
        "{}/test_results_mrpc.txt".format(base_folder), sep="\t")
    probs = torch.softmax(torch.from_numpy(prediction_mt), dim=1)

    # aggregate gold values
    if aggregate_predictions is None and keys is None:
        aggregate_predictions = {}
        keys = []

    predicted_pairs = set()
    for k, v_s in aggregate_predictions.items():
        for v in v_s:
            predicted_pairs.add((v[0], v[1]))

    for i in range(len(test_dat)):
        if prediction['prediction'][i] == 1:
            # H30-1-A Q0 886 1 0.193 JNLP
            query_id = test_dat["#1 ID"][i]
            c_id = test_dat["#2 ID"][i]
            score = probs[i][1]
            if query_id not in aggregate_predictions:
                keys.append(query_id)
                aggregate_predictions[query_id] = []

            if (query_id, c_id) not in predicted_pairs:
                aggregate_predictions[query_id].append((query_id, c_id, score))
                predicted_pairs.add((query_id, c_id))

    return aggregate_predictions, keys


def aggregate_all_results_task5(prediction_files, gold_test_file):
    prediction_mt = [pickle.load(open(f_, 'rb')) for f_ in prediction_files][0]

    # load test file - gold data for question id and article idß
    test_dat = pd.read_csv(gold_test_file, sep=",")

    predicted_pairs = {}

    probs = torch.softmax(torch.from_numpy(prediction_mt), dim=1)
    count_true = 0
    for i in range(len(test_dat)):
        # H30-1-A Q0 886 1 0.193 JNLP
        query_id = test_dat["id"][i]
        lb = test_dat["label"][i] == 1
        score = probs[i][1]
        predicted_pairs[query_id] = score

        pred_lb = True if  probs[i][1] > probs[i][0] else False

        if pred_lb == lb:
            count_true += 1
        
    print("acc={}, true={}, total={}".format(count_true / len(test_dat), count_true, len(test_dat)))

    return count_true / len(test_dat), (count_true, len(test_dat), predicted_pairs)

def aggregate_all_results(prediction_files, gold_test_file, topk=1, append_unpredicted_q=True, miss_ids_prediction_file=None):
    prediction_mt_s = [pickle.load(open(f_, 'rb')) for f_ in prediction_files]

    # load test file - gold data for question id and article idß
    test_dat = pd.read_csv(gold_test_file, sep=",")

    predicted_pairs = {}
    unpredicted_pairs = {}
    individual_model_stats = [[] for i in range(len(prediction_files))]

    for i_mod, prediction_mt in enumerate(prediction_mt_s):
        probs = torch.softmax(torch.from_numpy(prediction_mt), dim=1)

        for i in range(len(test_dat)):
            # H30-1-A Q0 886 1 0.193 JNLP
            query_id = test_dat["#1 ID"][i]
            c_id = re.sub(r'-sub\d+', '', test_dat["#2 ID"][i])
            score = probs[i][1]

            if probs[i][1] > probs[i][0]:
                if (query_id, c_id) not in predicted_pairs:
                    predicted_pairs[(query_id, c_id)] = []

                predicted_pairs[(query_id, c_id)].append(score)
            else:
                if (query_id, c_id) not in unpredicted_pairs:
                    unpredicted_pairs[(query_id, c_id)] = []

                unpredicted_pairs[(query_id, c_id)].append(score)

            # stats each model
            individual_model_stats[i_mod].append((query_id, c_id, score))

    # sort stats each model
    new_stats = [{} for i in range(len(prediction_files))]
    for i_mod, result in enumerate(individual_model_stats):
        for stat_e in result:
            if stat_e[0] not in new_stats[i_mod]:
                new_stats[i_mod][stat_e[0]] = []
            new_stats[i_mod][stat_e[0]].append((stat_e[1], stat_e[2].item()))
        for q_id, v in new_stats[i_mod].items():
            new_stats[i_mod][q_id].sort(key=lambda x: x[1], reverse=True)
            new_stats[i_mod][q_id] = new_stats[i_mod][q_id][:topk]
    individual_model_stats = new_stats

    #
    # aggregrate result from many models
    def aggregrate_result_(pairs_):
        aggregate_results = {}
        for k, v in pairs_.items():
            if k[0] not in aggregate_results:
                aggregate_results[k[0]] = []
            # aggregate_results[k[0]].append((k[0], k[1], max(v)))
            aggregate_results[k[0]].append((k[0], k[1], sum(v) / len(v)))
        return aggregate_results

    predicted_results = aggregrate_result_(predicted_pairs)
    unpredicted_results = aggregrate_result_(unpredicted_pairs)

    # append unpredicted question by top 1
    miss_prediction_keys = set()
    if append_unpredicted_q:
        miss_prediction_keys = set(unpredicted_results.keys()).difference(
            set(predicted_results.keys()))
        print('Miss {} question ids: {}'.format(len(miss_prediction_keys), miss_prediction_keys))
        if miss_ids_prediction_file is not None:
            json.dump(list(miss_prediction_keys), open(miss_ids_prediction_file, "wt", encoding='utf8'))
        for q_id in miss_prediction_keys:
            unpredicted_results[q_id].sort(key=lambda x: x[2], reverse=True)
            predicted_results[q_id] = unpredicted_results[q_id][:topk if topk is not None else 1]

    #
    # aggregrate gold label
    gold_results = {}
    gold_all_q_ids = set()
    for i in range(len(test_dat)):
        query_id = test_dat["#1 ID"][i]
        # test_dat["#2 ID"][i]
        c_id = re.sub(r'-sub\d+', '', test_dat["#2 ID"][i])
        gold_all_q_ids.add(query_id)

        if test_dat['label'][i] == 1:
            if query_id not in gold_results:
                gold_results[query_id] = []
            gold_results[query_id].append((query_id, c_id, 1))
    #
    # compute performance by accuracy task 4
    stats_task4 = {'pred': [], 'gold': []}
    for q_id in gold_all_q_ids:
        if q_id in gold_results:
            stats_task4['gold'].append((q_id, True))
        else:
            stats_task4['gold'].append((q_id, False))

        if q_id in predicted_results:
            stats_task4['pred'].append((q_id, True))
        else:
            stats_task4['pred'].append((q_id, False))
    right_count = len(set(stats_task4['pred']).intersection(
        set(stats_task4['gold'])))
    stats_task4['acc'] = right_count / len(gold_all_q_ids)
    stats_task4['correct_count'] = right_count
    stats_task4['total'] = len(gold_all_q_ids)

    #
    # compute performance by some metrics
    stats_result = {}
    for q_id in gold_all_q_ids:
        stats_result[q_id] = {}
        if q_id not in gold_results or q_id not in predicted_results:
            stats_result[q_id]['pred'] = [x[1]
                                          for x in predicted_results.get(q_id, [])]
            stats_result[q_id]['enssemble_score'] = [x[2].item()
                                                     for x in predicted_results.get(q_id, [])]
            stats_result[q_id]['gold'] = []
            stats_result[q_id]["P"] = 0
            stats_result[q_id]["R"] = 0
            stats_result[q_id]["F2"] = 0
        else:
            articles_prediction = [x[1]for x in predicted_results[q_id]]
            articles_gold = [x[1]for x in gold_results[q_id]]
            stats_result[q_id]['pred'] = articles_prediction
            stats_result[q_id]['enssemble_score'] = [x[2].item()
                                                     for x in predicted_results[q_id]]
            stats_result[q_id]['gold'] = articles_gold
            count_true = len(
                set(articles_prediction).intersection(set(articles_gold)))
            stats_result[q_id]["P"] = count_true / \
                len(set(articles_prediction))
            stats_result[q_id]["R"] = count_true / len(set(articles_gold))
            stats_result[q_id]["F2"] = f_score(
                stats_result[q_id]["P"],  stats_result[q_id]["R"], beta=2)

        stats_result[q_id]['found_by_model'] = q_id not in miss_prediction_keys
        stats_result[q_id]['detail_scores'] = [individual_model_stats[i][q_id]
                                               for i in range(len(prediction_files))]

    all_p = [stats_result[q_id]['P'] for q_id in stats_result]
    p = sum(all_p) / len(all_p)

    all_r = [stats_result[q_id]['R'] for q_id in stats_result]
    r = sum(all_r) / len(all_r)

    all_f2 = [stats_result[q_id]['F2'] for q_id in stats_result]
    macro_f2 = sum(all_f2) / len(all_f2)

    f2 = f_score(p, r, beta=2)

    overall_result = {'p': p, 'r': r, 'f2': f2,
                      'macro_f2': macro_f2, 'acc_task4': stats_task4}
    stats_result.update(overall_result)
    # pprint(stats_result)
    print('task 4:', "{:2.2f}".format(
        stats_task4['acc']*100), stats_task4['correct_count'], stats_task4['total'])

    return stats_result


def generate_file_submission(stats_result: Dict[str, Any], file_name: str, topk: int = None):
    predictions = {}
    for q_id, a_info in stats_result.items():
        if '-' not in q_id:
            continue
        if q_id not in predictions:
            predictions[q_id] = []
        if topk is None:
            for i_pred, pred in enumerate(zip(a_info['pred'], a_info['enssemble_score'])):
                predictions[q_id].append((q_id, pred[0], pred[1]))
        else:
            enssemble_scores = {}
            # aggregate all score
            for scores_model_i in a_info['detail_scores']:
                for score in scores_model_i:
                    a_id = score[0]
                    score_raw = score[1]
                    if a_id not in enssemble_scores:
                        enssemble_scores[a_id] = []
                    enssemble_scores[a_id].append(score_raw)
            # get mean all score
            for a_id in enssemble_scores:
                # max(enssemble_scores[a_id]) #
                enssemble_scores[a_id] = sum(
                    enssemble_scores[a_id]) / len(enssemble_scores[a_id])

            for a_id, score_enss in enssemble_scores.items():
                predictions[q_id].append((q_id, a_id, score_enss))

    keys_ = predictions.keys()
    for query_id in keys_:
        predictions[query_id].sort(key=lambda x: x[2], reverse=True)
        if topk is not None:
            # if len(predictions[query_id]) < topk:
            #     print("exception in {}, countpred = {}, topk={}".format(query_id, len(predictions[query_id]), topk))
            predictions[query_id] = predictions[query_id][:topk]
    prediction_str = []
    for query_id in keys_:
        for i, prediction_info in enumerate(predictions[query_id]):
            template = "{} {} {} {} {:.9f} {}"

            # H30-1-A Q0 886 1 0.193 JNLP
            prediction_str.append(
                template.format(query_id, "Q0", prediction_info[1], i + 1, prediction_info[2], "JNLP"))

    with open(file_name, "wt", encoding="utf8") as f:
        f.write("\n".join(prediction_str))
 