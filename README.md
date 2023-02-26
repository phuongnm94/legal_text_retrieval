# Coliee Task3

## Data
- structure of data directory (same structure between enlgish and japanese datasets)
    ```
        data/COLIEE2021statute_data-English/ 
        data/COLIEE2021statute_data-Japanese/
        ├── data_jp_topk_150_r02_r03 # folder save output of data_generator
        │   ├── dev.csv
        │   ├── stats.txt
        │   ├── test.csv
        │   ├── tfidf_classifier.pkl
        │   ├── all_data.json
        │   └── train.csv
        ├── text
        │   └── civil_code_jp-1to724-2.txt
        └── train
            ├── riteval_H18_en.xml
            ├── ...
            └── riteval_R01_en.xml
    ```

## Environments
```bash 
conda create -n env_coliee python=3.8
conda activate env_coliee
pip install -r requirements.txt
```

## All runs: 
1. Use vscode debugging for better checking runs: config in file `.vscode/launch.json`
2. **Runs**:
   1. **baseline model using BERT** 
      1. generate data, extract raw data from COLIEE competition to the `.json` and `.csv` data for training process: 
            
            ```bash
            conda activate env_coliee && cd src/ && python src/data_generator.py --path_folder_base data/COLIEE2023statute_data-Japanese/ --meta_data_alignment data/COLIEE2023statute_data-English/ --path_output_dir data/COLIEE2023statute_data-Japanese/data_ja_topk_150_r02_r03/ --lang jp --topk 150 --type_data task3 --dev_ids R02 --test_ids R03  
            ``` 
            Output (recall score is important)
            ```
            ...
            - 954 116400 1003 P:  0.008195876288659793 R:  0.9511465603190429 F1:  0.016251714181068626 F2:  0.03961399196093413  # eval train set using tfidf top 150
            Number data pairs:  116449
            - 128 16650 138 P:  0.0076876876876876875 R:  0.927536231884058 F1:  0.015248987371932332 F2:  0.03720497616556215  # eval valid set using tfidf top 150
            Number data pairs:  16650 
            - len(train_data_pairs_id), len(test_data_pairs_id), len(dev_data_pairs_id) =  116449 16650 16650
            ```
      2. train model by finetuning BERT or pretrained Japanese model (https://huggingface.co/cl-tohoku/bert-base-japanese-v2 or https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking )
            ```bash
            mkdir settings # folder save output 
            
            conda activate env_coliee && cd scripts && bash train_new.sh && cd ..
            ``` 
            or 
            ```
             mkdir settings && qsub scripts/train_new.sh
            ```
      3. infer and evaluation 
         - updating ... 
            <!-- ```bash
            # infer 
            conda activate coliee && cd scripts && bash run_infer_task3_basline.sh && cd ..
            
            # eval
            cd src && python evaluate.py --test_file ../data/COLIEE2021statute_data-Japanese/data_jp_topk_150_r02_r03/test.csv --pred_file ../data/settings/xlm-roberta-base/test_infer/test_results_None.txt --task task3 --append_unpredicted_q --run_name JNLP.CrossL 
            ```
            Output i saved in folder `data/settings/xlm-roberta-base/test_infer` including submission files and statistic files: 
            ```bash 
            ...
            # avg F2 (macroF2 *), F2 (based on macroP, macroR), macroP, macroR, return, retrieval-case
            68.8 - 69.93 - 68.62 - 70.27 - 92 - 72
            ...
            ```  -->

