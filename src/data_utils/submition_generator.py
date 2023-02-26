import pickle

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, accuracy_score
from data_utils.utils import aggregate_all_results, aggregate_results, generate_file_submission


def generate_result_task3():
    base_folder = "../coliee3_2020/test_prediction/"
    aggregate_predictions, keys = aggregate_results(base_folder)
    # base_folder2 = "../coliee3_2020/test_prediction/"
    # aggregate_predictions, keys = aggregate_results(base_folder2, aggregate_predictions, keys)
    generate_file_submission(aggregate_predictions, keys, "{}/JNLP.task3.tfidf-bert.txt".format(base_folder))

    # aggregate top 100 values
    topk = 100
    base_folder = "../coliee3_2020/test_prediction/"
    aggregate_predictions, keys = aggregate_all_results(base_folder)
    # base_folder2 = "../coliee3_2020/test_prediction/"
    # aggregate_predictions, keys = aggregate_all_results(base_folder2, aggregate_predictions, keys)
    generate_file_submission(aggregate_predictions,
                             keys,
                             "{}/JNLP.task3.tfidf-bert-L.txt".format(base_folder),
                             topk=100)


def _aggregate_result(predictions):
    predictions.sort(key=lambda x: x[3])
    return predictions[0][2]


def aggregate_results_task4(base_folder, aggregate_predictions=None, keys=None):
    prediction_mt = pickle.load(open("{}/predictions.pkl".format(base_folder), "rb"))
    test_dat = pd.read_csv("{}/test.tsv".format(base_folder), sep="\t")
    prediction = pd.read_csv("{}/test_results_mrpc.txt".format(base_folder), sep="\t")
    probs = torch.softmax(torch.from_numpy(prediction_mt), dim=1)

    # aggregate gold values
    if aggregate_predictions is None and keys is None:
        aggregate_predictions = {}
        keys = []

    for i in range(len(test_dat)):
        query_id = test_dat["#1 ID"][i]
        c_id = test_dat["#2 ID"][i]
        predict_lb = prediction['prediction'][i]
        gold_lb = None
        if "Quality" in test_dat:
            gold_lb = test_dat['Quality'][i]
        score = probs[i][predict_lb]
        if query_id not in aggregate_predictions:
            keys.append(query_id)
            aggregate_predictions[query_id] = []
        aggregate_predictions[query_id].append((query_id, c_id, predict_lb, score, gold_lb))

    # calculate the accuracy and print
    pred = []
    gold = []
    for query_id in keys:
        prediction_lb_ = 1 if 1 in [e[2] for e in aggregate_predictions[query_id]] else 0 #_aggregate_result(aggregate_predictions[query_id])
        pred.append(prediction_lb_)
        gold_lb_ = 1 if 1 in [e[4] for e in aggregate_predictions[query_id]] else 0
        gold.append(gold_lb_)
    print("confusion_matrix:\n", confusion_matrix(gold, pred))
    print("acc:\n", accuracy_score(gold, pred))

    return aggregate_predictions, keys


def generate_file_submission_task4(aggregate_predictions_, keys_, file_name, topk=None):
    for query_id in keys_:
        aggregate_predictions_[query_id].sort(key=lambda x: x[3], reverse=True)
        if topk is not None:
            aggregate_predictions_[query_id] = aggregate_predictions_[query_id][:topk]
    prediction_str = []
    for query_id in keys_:
        prediction_lb_ = 'Y' if 1 in [e[2] for e in aggregate_predictions_[query_id]] else 'N'

        # H30-1-A label JNLP
        template = "{} {} {}"
        prediction_str.append(template.format(query_id, prediction_lb_, "JNLP"))

    with open(file_name, "wt", encoding="utf8") as f:
        f.write("\n".join(prediction_str))


def generate_result_task4():
    base_folder = "../coliee4_2020/test_prediction_tfidfbert_negative/"
    aggregate_predictions, keys = aggregate_results_task4(base_folder)
    # base_folder2 = "../coliee4_2020/test_prediction_phrasetrans/"
    # aggregate_predictions, keys = aggregate_results_task4(base_folder2, aggregate_predictions, keys)
    generate_file_submission_task4(aggregate_predictions, keys, "{}/JNLP.task4.tfidf-bert.txt".format(base_folder))


if __name__ == "__main__":
    generate_result_task4()
