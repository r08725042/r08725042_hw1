#!/usr/bin/env bash

embedding_pkl=https://www.dropbox.com/s/05qbranjadpcteh/embedding.pkl?dl=1
extractive_lstm_pkl=https://www.dropbox.com/s/05qbranjadpcteh/embedding.pkl?dl=1
seq2seq_model_pkl=https://www.dropbox.com/s/x5huu77hh4rs8yj/seq2seq_model.pkl?dl=1
attention_model_pkl=https://www.dropbox.com/s/ofiahr0np7f954l/attention_model.pkl?dl=1

wget "${embedding_pkl}" -O ./embedding.pkl
wget "${extractive_lstm_pkl}" -O ./extractive_lstm.pkl
wget "${seq2seq_model_pkl}" -O ./seq2seq_model.pkl
wget "${attention_model_pkl}" -O ./attention_model.pkl