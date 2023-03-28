from data_utils import util_coliee
import pandas as pd
import json



def check_is_usecase(q: str):
    q = q.replace(',', ' ').replace('.', ' ').replace(')',
                                                      ' ').replace(')', ' ').replace("'", ' ')
    for i in range(1, len(q) - 1):
        if q[i-1] == ' ' and q[i+1] == ' ' and q[i].isupper() and q[i] not in ['A', 'I']:
            return True
    return False

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', default=None)
    
    parser.add_argument('--input_test', default='/home/phuongnm/coliee/data/COLIEE2023statute_data-English/train/riteval_R03_en.xml')
    parser.add_argument('--input_prediction', default='/home/phuongnm/coliee/settings/bert-base-japanese-whole-word-masking_5ckpt_150-newE5Seq512L2e-5/datout/test_R03.txt')
    parser.add_argument('--civi_code_path', default='/home/phuongnm/coliee/libs/statute_law_IR/data/parsed_civil_code/en_civil_code.json')
    parser.add_argument('--usecase_only', action='store_true', default=False)
    parser.add_argument('--log_correct_prediction', action='store_true', default=False)
    
    args = parser.parse_args()
    return args



def evaluate(INPUT_TEST, INPUT_PREDICTION, USECASE_ONLY, 
          print_out=True, 
          PARSED_CIVIL_CODE_PATH = None, log_correct_prediction=False):


    test_samples = util_coliee.load_samples(INPUT_TEST)
    result = dict()
    id2question = {}
    for s in test_samples:
        result[s.get('index')] = s.get('result')
        id2question[s.get('index')] = s.get('content')

    df = pd.read_csv(INPUT_PREDICTION, sep=' ', names=[
                    'id', 'Q0', 'art_id', 'rank', 'score', 'lang'])
    id2art = df.groupby('id')['art_id'].apply(list).to_dict()
    id2rank = df.groupby('id')['rank'].apply(list).to_dict()
    predicted = id2art

    f2_track = []
    returnn = 0
    retrieved = 0
    precision_track = []
    recall_track = []

    if log_correct_prediction:

        def get_json(path):
            with open(path) as f:
                ret = json.load(f)
            return ret
        
        civil_data = get_json(PARSED_CIVIL_CODE_PATH)
    res = {}
    for id, ans in result.items():
        question = id2question[id]
        if USECASE_ONLY and not check_is_usecase(question):
            continue

        if id not in predicted:
            f2_track.append(0)
            precision_track.append(0)
            recall_track.append(0)
            continue
        pred = set(predicted[id])
        correct_relevant = len([x for x in pred if x in ans])
        if log_correct_prediction:
            if correct_relevant:
                print('Query:')
                print(f"\tQuery ID: {id}")
                print(f"\tQuery: {question}")
                print(f"Correct predicted relevant article:")
                foo = [x for x in pred if x in ans]
                for ii, x in enumerate(foo):
                    print(f"\tCorrect prediction {ii+1}/{len(foo)}:")
                    print(f"\t\tArticle ID: {x}")
                    print(f"\t\tArticle: {civil_data[x]['content']}")
                print()

        returnn += len(pred)
        retrieved += correct_relevant

        precision = correct_relevant / len(pred)
        recall = correct_relevant / len(ans)
        if precision == 0 or recall == 0:
            f2 = 0
        else:
            f2 = (5 * precision * recall) / (4 * precision + recall) * 100
        f2_track.append(f2)
        precision_track.append(precision)
        recall_track.append(recall)
        res[id] = f2 
    json.dump(res, open('f2.json', "wt"))
    # print(f2_track)

    macro_f2 = round(sum(f2_track)/len(f2_track), 2)
    if print_out:
        print(f'Macro F2:', macro_f2)
        print(f'Return:', returnn, 'Retrieved:', retrieved, 'Precision:', round(100 * sum(precision_track) /
            len(precision_track), 2), 'Recall:', round(100 * sum(recall_track)/len(recall_track), 2))
    
    return macro_f2


def main(args, print_out=True):
    return evaluate(INPUT_TEST = args.input_test, INPUT_PREDICTION = args.input_prediction, USECASE_ONLY = args.usecase_only, 
                 PARSED_CIVIL_CODE_PATH=args.civi_code_path) 

if __name__ == '__main__':
    args = parse_args()
    main(args)
