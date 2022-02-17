for corpus in 'baseline' 'advadd'; do
     python inference/abstract_retrieval_tfidf.py \
         --corpus ../data/corpus_${corpus}.jsonl \
         --dataset ../data/claims_dev.jsonl \
         --k 3 \
         --min-gram 1 \
         --max-gram 2 \
         --output output/${corpus}/abstract_retrieval.jsonl

     python inference/rationale_selection_transformer.py \
          --corpus ../data/corpus_${corpus}.jsonl \
          --dataset ../data/claims_dev.jsonl \
          --abstract-retrieval output/${corpus}/abstract_retrieval.jsonl \
          --model model/rationale_roberta_large_scifact/ \
          --output-flex output/${corpus}/rationale_selection.jsonl

     python inference/label_prediction_transformer.py \
          --corpus ../data/corpus_${corpus}.jsonl \
          --dataset ../data/claims_dev.jsonl \
          --rationale-selection output/${corpus}/rationale_selection.jsonl \
          --model model/label_roberta_large_fever_scifact/ \
          --output output/${corpus}/label_prediction.jsonl

     python inference/merge_predictions.py \
          --rationale-file output/${corpus}/rationale_selection.jsonl \
          --label-file output/${corpus}/label_prediction.jsonl \
          --result-file output/${corpus}/merged_predictions.jsonl

     python evaluate/pipeline.py \
          --gold ../data/claims_dev.jsonl \
          --corpus ../data/corpus_${corpus}.jsonl \
          --prediction output/${corpus}/merged_predictions.jsonl \
          --output output/${corpus}/metrics.json
done