# setup
./download-model.sh rationale roberta_large scifact
./download-model.sh label roberta_large fever_scifact

mkdir output
mkdir output/baseline
mkdir output/advadd

# baseline
python inference/abstract_retrieval_tfidf.py \
    --corpus ../data/corpus_baseline.jsonl \
    --dataset ../data/claims_dev.jsonl \
    --k 3 \
    --min-gram 1 \
    --max-gram 2 \
    --output output/baseline/abstract_retrieval.jsonl

python inference/rationale_selection_transformer.py \
     --corpus ../data/corpus_baseline.jsonl \
     --dataset ../data/claims_dev.jsonl \
     --abstract-retrieval output/baseline/abstract_retrieval.jsonl \
     --model model/rationale_roberta_large_scifact/ \
     --output-flex output/baseline/rationale_selection.jsonl

python inference/label_prediction_transformer.py \
     --corpus ../data/corpus_baseline.jsonl \
     --dataset ../data/claims_dev.jsonl \
     --rationale-selection output/baseline/rationale_selection.jsonl \
     --model model/label_roberta_large_fever_scifact/ \
     --output output/baseline/label_prediction.jsonl

python inference/merge_predictions.py \
     --rationale-file output/rationale_selection.jsonl \
     --label-file output/baseline/label_prediction.jsonl \
     --result-file output/baseline/merged_predictions.jsonl

python evaluate/pipeline.py \
     --gold ../data/claims_dev.jsonl \
     --corpus ../data/corpus_baseline.jsonl \
     --prediction output/baseline/merged_predictions.jsonl \
     --output output/baseline/metrics.json


# advadd
python inference/abstract_retrieval_tfidf.py \
    --corpus ../data/corpus_advadd.jsonl \
    --dataset ../data/claims_dev.jsonl \
    --k 3 \
    --min-gram 1 \
    --max-gram 2 \
    --output output/advadd/abstract_retrieval.jsonl

python inference/rationale_selection_transformer.py \
     --corpus ../data/corpus_advadd.jsonl \
     --dataset ../data/claims_dev.jsonl \
     --abstract-retrieval output/advadd/abstract_retrieval.jsonl \
     --model model/rationale_roberta_large_scifact/ \
     --output-flex output/advadd/rationale_selection.jsonl

python inference/label_prediction_transformer.py \
     --corpus ../data/corpus_advadd.jsonl \
     --dataset ../data/claims_dev.jsonl \
     --rationale-selection output/advadd/rationale_selection.jsonl \
     --model model/label_roberta_large_fever_scifact/ \
     --output output/advadd/label_prediction.jsonl

python inference/merge_predictions.py \
     --rationale-file output/advadd/rationale_selection.jsonl \
     --label-file output/advadd/label_prediction.jsonl \
     --result-file output/advadd/merged_predictions.jsonl

python evaluate/pipeline.py \
     --gold ../data/claims_dev.jsonl \
     --corpus ../data/corpus_advadd.jsonl \
     --prediction output/advadd/merged_predictions.jsonl \
     --output output/advadd/metrics.json
