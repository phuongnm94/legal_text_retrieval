import json
import os

from tfidf_classifier import do_classify
from utils import load_data_kse


def do_generate_stopwords(path_folder_base="../coliee3_2020/", threshold=0.02, tokenizer=None, data=None):
    c_docs, c_keys, dev_q, test_q, train_q = load_data_kse(
        path_folder_base=path_folder_base) if data is None else data

    _, (test_similarities, c_vect, test_q_vect, vectorizer) = do_classify(c_docs, c_keys, test_q,
                                                                          stopwords_=None,
                                                                          topk=150,
                                                                          tokenizer=tokenizer)
    # generate stop words
    stop_words_idx = []
    for doc_vect in c_vect:
        for word_idx in doc_vect.indices:
            if doc_vect[0, word_idx] < threshold and word_idx not in stop_words_idx:
                stop_words_idx.append(word_idx)

    stop_words = [vectorizer.get_feature_names()[w_idx]
                  for w_idx in stop_words_idx]
    path_folder_data_out = "{}/stopwords/".format(path_folder_base)
    if not os.path.exists(path_folder_data_out):
        os.mkdir(path_folder_data_out)
    json.dump(stop_words, open(
        "{}/stopwords.json".format(path_folder_data_out), "wt"), ensure_ascii=False)


if __name__ == "__main__":
    do_generate_stopwords()
