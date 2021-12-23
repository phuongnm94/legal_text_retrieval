#!/usr/bin/env python
# coding: utf-8
import json
import pickle

from transformers.tokenization_utils import PreTrainedTokenizer
from data_generator import vi_tokenize

from tfidf_classifier import do_classify
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from transformers.data.datasets.glue import *
from transformers.data.processors.utils import InputExample

from utils import Question, load_data_kse, standardize_data, Article
import numpy as np


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


class LawDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """
    args: GlueDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
            self,
            args: GlueDataTrainingArguments,
            tokenizer: PreTrainedTokenizer,
            limit_examples: Optional[int] = None,
            mode: Union[str, Split] = Split.train,
            c_code=None,
            sentence=None,
    ):
        self.args = args
        task_name = 'mrpc'
        self.processor = glue_processors[task_name]()
        self.output_mode = 'classification'
        self.c_code = c_code if c_code is not None else []
        self.sentence = sentence if sentence is not None else ""
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")

        label_list = self.processor.get_labels()
        self.label_list = label_list

        def _create_examples(lines, set_type='test'):
            examples = []
            for (i, line) in enumerate(lines):
                guid = "%s-%s" % (set_type, i)
                text_a = line[3]
                text_b = line[4]
                label = None if set_type == "test" else line[0]
                examples.append(InputExample(
                    guid=guid, text_a=text_a, text_b=text_b, label=label))
            return examples

        lines = []
        for i, e in enumerate(self.c_code):
            lines.append([0, "sent_{}".format(
                i), e[1], self.sentence[i], e[0]])

        # recreate the data
        examples = _create_examples(lines)
        if limit_examples is not None:
            examples = examples[:limit_examples]
        self.features = glue_convert_examples_to_features(
            examples,
            tokenizer,
            max_length=args.max_seq_length,
            label_list=label_list,
            output_mode=self.output_mode,
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list

    def get_c_code_ids(self):
        return [e[1] for e in self.c_code]


def infer_coliee_task3(sentence, all_civil_code, data_args, tfidf_vectorizer, trainer, bert_tokenizer,  tokenizer=None, topk=150):
    if isinstance(sentence, str):
        sentence = [sentence]
    test_q = [Question(id='q{}'.format(i), content=tokenizer(
        s)if tokenizer is not None else s, content_raw=s, relevant_a=[]) for i, s in enumerate(sentence)]
    c_docs = all_civil_code[0]
    c_keys = all_civil_code[1]
    c_vect = all_civil_code[2]
    c_docs_keys = list(zip(all_civil_code[0], all_civil_code[1]))

    test_pred, _ = do_classify(
        c_docs, c_keys, test_q, vectorizer=tfidf_vectorizer, topk=topk, c_vect=c_vect,  combine_score=(tokenizer is None))

    c_code_pred_by_tfidf = []
    coressponding_questions = []
    for i, s_pred in enumerate(test_pred):
        for idx in s_pred:
            coressponding_questions.append(test_q[i].content)
            c_code_pred_by_tfidf.append(c_docs_keys[idx])

    test_dataset = LawDataset(data_args,
                              bert_tokenizer,
                              mode='test', sentence=coressponding_questions, c_code=c_code_pred_by_tfidf)
    predictions = trainer.predict(test_dataset=test_dataset).predictions
    probs = torch.softmax(torch.from_numpy(predictions), dim=1)
    predicted_labels = torch.argmax(probs, 1)
    return predicted_labels, probs, c_code_pred_by_tfidf

def list_split(listA, n):
    for x in range(0, len(listA), n):
        every_chunk = listA[x: n+x]

        if len(every_chunk) < n:
            every_chunk = every_chunk + \
                [None for y in range(n-len(every_chunk))]
        yield every_chunk 

def init_state(path_c_code, path_data_org, path_preprocessed_data, model_path, tokenizer=None, topk=150, testing_data=None, max_seq_length=512,
               do_lower_case=True):
    model_version = model_path  # 'bert-base-uncased'

    config = AutoConfig.from_pretrained(
        model_version,
        num_labels=2,
        finetuning_task='MRPC'
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_version, config=config)
    bert_tokenizer = AutoTokenizer.from_pretrained(
        model_version, do_lower_case=do_lower_case)
    model.eval()

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args=["--model_name_or_path", model_version,
              "--task_name", "MRPC",
              "--data_dir", "./coliee3_2020/data",
              "--do_predict",
              "--per_device_train_batch_size", "16",
              "--max_seq_length", "{}".format(max_seq_length),
              "--learning_rate", "2e-5",
              "--output_dir", model_version,
              "--overwrite_output_dir"])
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args
    )
    tfidf_vectorizer = pickle.load(
        open("{}/tfidf_classifier.pkl".format(path_preprocessed_data), "rb"))
    if isinstance(tfidf_vectorizer, tuple):
        tfidf_vectorizer, bm25_scorer = tfidf_vectorizer[0], tfidf_vectorizer[1]

    path_data_cached = '{}/tokenized_data_cached.pkl'.format(
        path_preprocessed_data)
    if os.path.isfile(path_data_cached):
        print("Load cached file data: {}".format(path_data_cached))
        c_docs, c_keys, dev_q, test_q, train_q, sub_doc_info = pickle.load(
            open(path_data_cached, 'rb'))
    else:
        print("Load data and tokenize data")
        c_docs, c_keys, dev_q, test_q, train_q, sub_doc_info = load_data_kse(
            path_folder_base=path_data_org,  ids_test=[
            ], tokenizer=tokenizer, testing_data=testing_data,
            # chunk_content_info=chunk_content_info
        )

    c_vect = tfidf_vectorizer.transform([standardize_data(d) for d in c_docs])
    return (c_docs, c_keys, c_vect), data_args, (tfidf_vectorizer, bm25_scorer), trainer, bert_tokenizer


if __name__ == "__main__":
    global all_civil_code, data_args, tfidf_vectorizer, trainer, bert_tokenizer

    model_configs = {
        # 'NlpHUST': {
        #     "path_data_org": 'data/zac2021-ltr-data/',
        #     "path_c_code": 'data/zac2021-ltr-data/legal_corpus.json',
        #     "tokenizer": None,
        #     "topk": 150,
        #     "do_lower_case": False,
        #     "max_seq_length": 512,
        #     "path_preprocessed_data": 'data/zalo-tfidfngrbm25150-notok-full/',
        #     "model_path": 'settings/NlpHTfbmngr150E5-notok-full42/models',
        # },
        'PhoBERT': {
            "path_data_org": 'data/zac2021-ltr-data/',
            "path_c_code": 'data/zac2021-ltr-data/legal_corpus.json',
            "tokenizer": 'vi_tokenize',
            "topk": 300,
            "do_lower_case": True,
            "max_seq_length": 256,
            "path_preprocessed_data": 'data/zalo-tfidfbm25150-full/',
            "model_path": 'settings/Tfbm150E5-full42/models',
        }
    }
    print(json.dumps(model_configs, indent=2))

    test_all_data = json.load(open('data/zac2021-ltr-data/public_test_question.json'))['items']
    test_ids = [e['question_id'] for e in test_all_data]
    test_sents = [e['question'] for e in test_all_data]

    # test_sents = [
    #     "Đới khoáng hóa là gì?",
    #     "Kinh phí bảo đảm thi hành án đối với pháp nhân thương mại được quy định như thế nào?",
    #     "Thời gian viên chức nghỉ thai sản có đánh giá chất lượng không?",
    #     "Việc trình, giải quyết hồ sơ đề nghị sửa đổi, bổ sung Quyết định giao khu vực biển được quy định như thế nào?",
    #     # "Hình thức kỷ luật hạ bậc lương trong việc xử lý VPHC sẽ áp dụng cho đối tượng nào?",
    #     # "Nguyên tắc xác định tổ chức, cá nhân làm môi trường bị ô nhiễm, suy thoái theo quy định của pháp luật",
    #     # "Người được đề xuất hình thức kỷ luật trong quốc phòng?",
    # ]

    # def init models
    model_init_states = {}
    print("Loading model ....")
    for m_name, model_info in model_configs.items():
        if 'tokenizer' in model_info and model_info['tokenizer'] == 'vi_tokenize':
            model_info['tokenizer'] = vi_tokenize

        model_init_states[m_name] = init_state(**model_info)
        all_civil_code, data_args, tfidf_vectorizer, trainer, bert_tokenizer = model_init_states[
            m_name]

        tokenizer = model_info.get('tokenizer')
        topk = 150
        infer_coliee_task3(sentence=test_sents[:5],
                           all_civil_code=all_civil_code, data_args=data_args, tfidf_vectorizer=tfidf_vectorizer,
                           trainer=trainer, bert_tokenizer=bert_tokenizer,
                           tokenizer=tokenizer, topk=topk)
    print("Finish loaded model")

    missing_ids_info = {}
    real_prediction = {}

    # start infer
    time_start = time.time()
    for m_name, model_info in model_configs.items():
        if 'tokenizer' in model_info and model_info['tokenizer'] == 'vi_tokenize':
            model_info['tokenizer'] = vi_tokenize

        all_civil_code, data_args, tfidf_vectorizer, trainer, bert_tokenizer = model_init_states[
            m_name]
        tokenizer = model_info.get('tokenizer')
        topk = model_info.get('topk', 150)
        predicted_labels, probs, c_code_pred_by_tfidf = infer_coliee_task3(sentence=test_sents, all_civil_code=all_civil_code,
                                                                           data_args=data_args,
                                                                           tfidf_vectorizer=tfidf_vectorizer,
                                                                           trainer=trainer, bert_tokenizer=bert_tokenizer,
                                                                           tokenizer=tokenizer, topk=topk)

        predicted_labels = [x for x in list_split(predicted_labels, topk)] # np.array_split(predicted_labels, len(test_sents))
        probs = [x for x in list_split(probs, topk)] #np.array_split(probs, len(test_sents))
        c_code_pred_by_tfidf = [x for x in list_split(c_code_pred_by_tfidf, topk)] # np.array_split( c_code_pred_by_tfidf, len(test_sents))

        result = [[{"label": True if lb == 1 else False,
                    "scores": [float(probs[jj][i][j]) for j in range(probs[jj][i].shape[0])],
                  "id": test_ids[jj],
                    "sentence": s,
                    #  "civil_code": c_code_pred_by_tfidf[jj][i][0],
                    "civil_code_id": c_code_pred_by_tfidf[jj][i][1],
                    }
                   for i, lb in enumerate(predicted_labels[jj]) if lb == 1] for jj, s in enumerate(test_sents)]

        current_missing_ids = [[{"label":  False,
                                      "score": float(probs[jj][i][1]),
                                      "id": test_ids[jj],
                                      "civil_code_id": c_code_pred_by_tfidf[jj][i][1],
                                      }
                                     for i, lb in enumerate(predicted_labels[jj]) if lb == 0] for jj, s in enumerate(test_sents)]
        for negative_prediction in current_missing_ids:
            negative_prediction.sort(key=lambda info: info['score'], reverse=True)

        missing_ids_info[m_name] = current_missing_ids
        
        for jj, k in enumerate(test_ids):
            if k not in real_prediction:
                real_prediction[k] = set()
            real_prediction[k] = real_prediction[k].union(
                set([pred_infor['civil_code_id'] for pred_infor in result[jj]]))

        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("Finish inference on fine-tuned model {}, total time consuming: ".format(
            m_name), time.time() - time_start)
        print(len(result))

    count_negative_add = 0
    for jj, k in enumerate(test_ids):
        if len(real_prediction[k]) == 0:
            count_negative_add += 1
            # pick 1 best score from negative prediction each model
            for m_name, _ in model_configs.items():
                real_prediction[k].add(missing_ids_info[m_name][jj][0]['civil_code_id'])

    print("Total time consuming for {} samples: {} seconds => avg 1 sample in {} second".format(
        len(test_sents), time.time() - time_start,  (time.time() - time_start) / len(test_sents)))

    submit_result = []
    for k, v in real_prediction.items():
        relevant_a_s = []
        for relevant_a in v:
            tmp_a = Article.from_string(relevant_a)
            relevant_a_s.append({'law_id': tmp_a.l_id, 'article_id': tmp_a.a_id})
        submit_result.append({
            'question_id': k,
            'relevant_articles': relevant_a_s
        })
    print("Count negative addition = {}".format(count_negative_add))

    json.dump(submit_result, open("data/result_prediction.json", "wt", encoding='utf8'), ensure_ascii=False, indent=2)


