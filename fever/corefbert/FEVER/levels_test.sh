for FILE in ../../data/add_sent/corefbert_input/*.*
do
echo $FILE
# echo $FILE > report_levels_trials.txt
# python test.py --outdir ../../data/add_sent/corefbert_output/ --test_path $FILE --bert_pretrain ../roberta_large --checkpoint ../checkpoint/kgat/model.best.pt --name temp.json --bert_hidden_dim 1024
# python batch_scoring_trials.py --filename $FILE
python3 test.py  --model_type roberta  --outdir output/ --test_path $FILE --bert_pretrain  ../pretrained_downloads/CorefRoBERTa-large  --batch_size 32 --checkpoint FEVER_CorefRoBERTa-large_devbert/model.best.pt --name temp.jsonl
python eval.py --base output/roberta_large_devbert_original.jsonl --new output/temp.jsonl
done
