
import argparse
import json
import pandas as pd
from typing import List
import math


def group_by_qid(question_ids, c_ids, relevants, scores):
    results = {}
    for i, question_id in enumerate(question_ids):
        if question_id not in results:
            results[question_id] = {'pred_c_ids': [], 'question_id': question_id, 'f2': 0.0, 'pred_c_scores':[], 'rank':[], 'topk': []}
        if bool(relevants[i]):
            results[question_id]['pred_c_ids'].append(c_ids[i])
            results[question_id]['pred_c_scores'].append(scores[i])

        results[question_id]['rank'].append((c_ids[i], scores[i]))

    for question_id in results:
        results[question_id]['rank'].sort(key=lambda x: x[1], reverse=True)

    return results
def generate_file_submission(stats_result, file_name: str, key_cids="pred_c_ids", key_scores="pred_c_scores", limited_prediction=None):
    prediction_str = []
    for q_id, a_info in stats_result.items(): 
        retrieve_info = list(zip(a_info[key_cids], a_info[key_scores]))
        retrieve_info.sort(key=lambda x: x[1], reverse=True)
        if limited_prediction is not None:
            retrieve_info = retrieve_info[:limited_prediction]

        for i, (pred_c_id, pred_c_score) in enumerate(retrieve_info):
            template = "{} {} {} {} {:.9f} {}"

            # H30-1-A Q0 886 1 0.193 JNLP
            prediction_str.append(
                template.format(q_id, "Q0", pred_c_id, i + 1, pred_c_score, "CAPTAIN"))

    with open(file_name, "wt", encoding="utf8") as f:
        f.write("\n".join(prediction_str))
    return len(prediction_str)

def enssemble_prediction(path_main: str, path_addition: List[str], key_cids="pred_c_ids", key_scores="pred_c_scores", 
                         addition_limit=None, relevant_limit=None):
    if relevant_limit is None:
        relevant_limit = 100000
    df_data = pd.read_csv(open(f'{path_main}'), sep=" ", header=None)
    gr_pred = group_by_qid(df_data[0].values, df_data[2].values, [1]*len(df_data[4].values), [math.e**score for score in df_data[4].values])
    for i, p_addition in enumerate(path_addition):
        df_addition = pd.read_csv(open(f'{p_addition}'), sep=" ", header=None)
        gr_p = group_by_qid(df_addition[0].values, df_addition[2].values, [1]*len(df_addition[4].values), df_addition[4].values)

        for k, v in gr_p.items():
            if len(gr_pred[k][key_cids]) > relevant_limit:
                gr_pred[k][key_cids] = gr_pred[k][key_cids][:relevant_limit]
                gr_pred[k][key_scores] = gr_pred[k][key_scores][:relevant_limit]

            candidates = list(zip(v[key_cids], v[key_scores]))
            candidates.sort(key= lambda x: x[1], reverse=True) 
            if addition_limit is not None:
                candidates = candidates[:addition_limit]

            for c_id, c_score in candidates:
                if c_id not in gr_pred[k][key_cids] and len(gr_pred[k][key_cids])<relevant_limit:
                    gr_pred[k][key_cids].append(c_id)
                    gr_pred[k][key_scores].append(c_score)

    return gr_pred


def enss():
    for id_check in ["02", "03"][-1:]:
        # path_main = f"/home/phuongnm/coliee/settings/bert-base-japanese-whole-word-masking_5ckpt_150-newE5Seq512L2e-5/datout/test_R{id_check}.txt"
        # path_main = f"/home/phuongnm/coliee/settings/bert-base-japanese-whole-word-masking_5ckpt_150-newE5Seq512L2e-5/datout/test_p844_10_{id_check}.txt"
        # path_main = f"/home/phuongnm/coliee/settings/bert-base-japanese-whole-word-masking_5ckpt_150-newE5Seq512L2e-5/datout/test_R{id_check}_5_50_009.txt"
        path_main = f"/home/phuongnm/coliee/settings/bert-base-japanese-whole-word-masking_5ckpt_150-newE5Seq512L2e-5/datout/test_R{id_check}_5_80_0015.txt"
        path_addition = [ 
            # f"/home/phuongnm/coliee/settings/bert-base-japanese-whole-word-masking_5ckpt_150-newE5Seq512L2e-5/datout/test_R{id_check}.txt",
            # f"/home/phuongnm/coliee/settings/bert-base-japanese-whole-word-masking_5ckpt_150-newE5Seq512L2e-5/datout/test_p844_10_{id_check}.txt",
            # f"/home/phuongnm/coliee/settings/bert-base-japanese-whole-word-masking_5ckpt_150-newE5Seq512L2e-5/datout/test_R{id_check}_5_50_009.txt",
            # f"/home/phuongnm/coliee/settings/bert-base-japanese-whole-word-masking_5ckpt_150-newE5Seq512L2e-5/datout/test_p844_10_{id_check}.txt",
            f"/home/phuongnm/coliee/settings/bert-base-japanese-whole-word-masking_5ckpt_150-newE5Seq512L2e-5/CAPTAIN.allEnss.tsv_{id_check}"
                        ]
        gr_pred = enssemble_prediction(path_main, path_addition)

        print(generate_file_submission(gr_pred, f'test{id_check}.txt'))


        # print(f"\nPred: {path_main}")
        # !python /home/phuongnm/coliee/libs/statute_law_IR/evaluate.py \
        #     --input_test /home/phuongnm/coliee/data/COLIEE2023statute_data-English/train/riteval_R{id_check}_en.xml \
        #     --input_prediction {path_main}

        # print(f"\nPred: enssemble")
        # !python /home/phuongnm/coliee/libs/statute_law_IR/evaluate.py \
        #     --input_test /home/phuongnm/coliee/data/COLIEE2023statute_data-English/train/riteval_R{id_check}_en.xml \
        #     --input_prediction /home/phuongnm/coliee/settings/bert-base-japanese-whole-word-masking_5ckpt_150-newE5Seq512L2e-5/models/test{id_check}.txt

if __name__=="__main__":

    # training+model args
    parser = argparse.ArgumentParser(description="Training Args")
    
        
    parser.add_argument("--file_output_id", type=str, default="allEnss", help="Id of submission")
    parser.add_argument("--file_pred", type=str, default=".", help="log dir")

    opts = parser.parse_args()
    generate_file_submission(json.load(open(opts.file_pred)), f"CAPTAIN.{opts.file_output_id}.tsv")
