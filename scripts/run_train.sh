cd /app && mkdir settings
cd /app && mkdir data 
USER=root

# generate data 
cd /app && mkdir data/zalo-tfidfbm25150-full
cd /app && python3 src/data_generator.py --path_folder_base data/zac2021-ltr-data/ --test_file public_test_question.json --topk 150 --type_data task3 --tok --path_output_dir data/zalo-tfidfbm25150-full


# train phoBERT 
cd /app/scripts && bash run_finetune_bert.sh $USER  vinai/phobert-base  ../  data/zalo-tfidfbm25150-full Tfbm150E5-full 5


# generate data 
cd /app && mkdir data/zalo-tfidfngrbm25150-notok-full
cd /app && python3 src/data_generator.py --path_folder_base data/zac2021-ltr-data/ --test_file public_test_question.json --topk 150 --type_data task3 --path_output_dir data/zalo-tfidfngrbm25150-notok-full

# train NLPHust 
cd /app/scripts && bash run_finetune_bert.sh $USER  NlpHUST/electra-base-vn  ../  data/zalo-tfidfngrbm25150-notok-full NlpHTfbmngr150E5-notok-full 5
