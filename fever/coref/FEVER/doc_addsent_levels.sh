for FILE in /dfs/user/yibingdu/fever-resources/data/add_sent/before_sent_retrieval/

python3 test.py  --model_type roberta  --outdir /dfs/user/yibingdu/fever-resources/data/add_sent/before_sent_retrieval/ --test_path $FILE --bert_pretrain ../pretrained_downloads/CorefRoBERTa-large --batch_size 32 --checkpoint FEVER_CorefRoBERTa-large_devbert/model.best.pt --name corefbert_.jsonl

