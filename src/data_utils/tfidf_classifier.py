import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_utils.utils import evaluate, load_data_coliee


def do_classify(c_docs_, c_keys_, test_q_, stopwords_=None, topk=150, vectorizer=None, tokenizer=None):
    if vectorizer is None:
        print("[W] Learning Tfidf Vectorizer ...")
        vectorizer = TfidfVectorizer(stop_words=stopwords_, tokenizer=tokenizer)
        vectorizer.fit(c_docs_)
    c_vect = vectorizer.transform(c_docs_)

    test_q_docs = [q["content"] for q in test_q_]
    test_q_vect = vectorizer.transform(test_q_docs)
    test_similarities = cosine_similarity(test_q_vect, c_vect)
    test_pred = evaluate(test_similarities, test_q_, topk=topk, c_keys=c_keys_)
    return test_pred, (test_similarities, c_vect, test_q_vect, vectorizer)

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from tqdm import tqdm
# from custom_rank_bm25 import BM25Plus
# import numpy as np

# from utils2 import Article, combine_idxs, evaluate, evaluate_idx, load_data_kse, standardize_data
# import pickle, json
 

# def do_classify(c_docs_, c_keys_, test_q_, stopwords_=None, topk=150, vectorizer=None, tokenizer=None, combine_score=False, c_vect=None):
#     # check old system 
#     if vectorizer is not None and isinstance(vectorizer, TfidfVectorizer):
#         return do_classify_old(c_docs_, c_keys_, test_q_, stopwords_=stopwords_, topk=topk, vectorizer=vectorizer, tokenizer=tokenizer)

#     # new system 
#     c_docs_ = [standardize_data(d) for d in c_docs_]
#     if vectorizer is None:
#         print("[W] Learning Tfidf Vectorizer ...")
#         tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords_, tokenizer=tokenizer, ngram_range=(1,2))
#         tfidf_vectorizer.fit(c_docs_)

#         print("[W] Learning BM25 Vectorizer ...")
#         bm25_scorer = BM25Plus([d.split(" ") for d in c_docs_])

#         vectorizer = (tfidf_vectorizer, bm25_scorer) 
#     else:
#         tfidf_vectorizer, bm25_scorer = vectorizer[0], vectorizer[1]

#     # get cosin score from tfidf vector
#     if c_vect is None:
#         c_vect = tfidf_vectorizer.transform(c_docs_)
#     test_q_docs = [standardize_data(q["content"]) for q in test_q_]
#     test_q_vect = tfidf_vectorizer.transform(test_q_docs)
#     tfidf_cosine_score = cosine_similarity(test_q_vect, c_vect)

#     # get bm25 score
#     bm25_similarities = []
#     for query in tqdm(test_q_docs):
#         bm25_similarities.append(bm25_scorer.get_scores(query.split(" ")))
#     bm25_similarities = np.array(bm25_similarities)

#     # combine score
#     if tokenizer is None:
#         final_score = 0.33*tfidf_cosine_score+ 0.67*bm25_similarities/np.max(bm25_similarities)
#         preds  = evaluate(final_score, test_q_, topk=topk, c_keys=c_keys_)
#     else:
#         idx_tfidf = tfidf_cosine_score.argsort()[:, ::-1][:, :topk]
#         idx_bm25 = bm25_similarities.argsort()[:, ::-1][:, :topk]
#         preds = combine_idxs(idx_tfidf, idx_bm25, topk)
#         preds  = evaluate_idx(preds, test_q_, c_keys=c_keys_)
    
#     return preds, ((tfidf_cosine_score, bm25_similarities), c_vect, test_q_vect, vectorizer)



# def do_classify_old(c_docs_, c_keys_, test_q_, stopwords_=None, topk=150, vectorizer=None, tokenizer=None):
#     if vectorizer is None:
#         print("[W] Learning Tfidf Vectorizer ...")
#         vectorizer = TfidfVectorizer(stop_words=stopwords_, tokenizer=tokenizer)
#         vectorizer.fit(c_docs_)
#     c_vect = vectorizer.transform(c_docs_)

#     test_q_docs = [q["content"] for q in test_q_]
#     test_q_vect = vectorizer.transform(test_q_docs)
#     test_similarities = cosine_similarity(test_q_vect, c_vect)
#     test_pred = evaluate(test_similarities, test_q_, topk=topk, c_keys=c_keys_)
#     return test_pred, (test_similarities, c_vect, test_q_vect, vectorizer)




# if __name__ == "__main__":
#     path_folder_base = "../coliee3_2020/"
#     c_docs, c_keys, dev_q, test_q, train_q, _ = load_data_coliee(path_folder_base=path_folder_base)

#     # stopwords = []
#     # stopwords = json.load(open("{}/stopwords/stopwords_.json".format(path_folder_base), "rt"))["words"]
#     stopwords = json.load(open("{}/stopwords/stopwords.json".format(path_folder_base), "rt"))
#     print(do_classify(c_docs, c_keys, test_q, topk=2, stopwords_=stopwords))
