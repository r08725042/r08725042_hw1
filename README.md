# ADL HW1  

(使用Python 3.6)

1. 首先先將train.jsonl, valid.jsonl, test.jsonl放入 r08725042/data 的資料夾中(其他位置亦可，predict model並未寫死路徑，而是由使用者輸入data位置)

2. 執行 'download.sh'，以下載各模型已訓練好的參數及embedding檔案

3. 執行 'install_packages.sh'，將需要的package下載 (此為助教提供，學生並未另外使用 'requirements.txt'內以外的任何package)

4. 使用Model

    - **如果"不重新"train model，直接進行預測**
        
        1. 執行 'extractive.sh' 並輸入test data的位置及檔案輸出位置，此部分會包含 test data的前處理，執行完畢後將output出預測結果

        2. 執行 'seq2seq.sh' 並輸入test data的位置及檔案輸出位置，此部分會包含 test data的前處理，執行完畢後將output出預測結果

        3. 執行 'attention.sh' 並輸入test data的位置及檔案輸出位置，此部分會包含 test data的前處理，執行完畢後將output出預測結果


    - **如果要重新train model**
        1. 分別執行 'preprocess_seq_tag.py' 以及 'preprocess_seq2seq.py' ，此時會產生六個檔案於該目錄
            - 'extractive_train.pkl'
            - 'extractive_valid.pkl'
            - 'extractive_test.pkl'
            - 'abstractive_train.pkl'
            - 'abstractive_valid.pkl'
            - 'abstractive_test.pkl'
            
        2. 執行 'extractive_train.py'，當Model訓練完畢後則會儲存 'extractive_lstm.pkl' ，用於後續預測用
        
        3. 執行 'seq2seq_train.py'，當Model訓練完畢後則會儲存 'seq2seq_model.pkl' ，用於後續預測用

        4. 執行 'attention_train.py'，當Model訓練完畢後則會儲存 'attention_model.pkl' ，用於後續預測用

5. 其他檔案說明
    - 資料前處理 : 
        - preprocess_seq2seq.py
        - preprocess_seq_tag.py
        - utils.py
        - dataset.py
        - seq2seqconfig.json
        - seq2tagconfig.json
    - extractive plot:
        - plot_extractive_result.py
    - attention plot:
        - plot_attention_weight.py