for corpus in 'baseline' 'advadd'; do
    #Eval abstract selection
    python ../abstract_retrieval/evaluate.py \
        --dataset ../../data/claims_dev.jsonl \
        --abstract-retrieval ../output/${corpus}_abstract_rerank_dev_mlm.jsonl

    #Eval rationale selection
    python ../rationale_selection/evaluate.py \
        --corpus ../../data/corpus_${corpus}.jsonl \
        --dataset ../../data/claims_dev.jsonl \
        --rationale-selection ../output/${corpus}_rationale_selection_dev_scibert_mlm.jsonl

    #Eval claim prediction selection
    PYTHONPATH=.. python ../pipeline/evaluate_paper_metrics.py \
        --dataset ../../data/claims_dev.jsonl \
        --corpus ../../data/corpus_${corpus}.jsonl \
        --rationale-selection  ../output/${corpus}_rationale_selection_dev_scibert_mlm.jsonl \
        --label-prediction ../output/${corpus}_kgat_dev_roberta_large_mlm_pred.jsonl

    python ../pipeline/merge_predictions.py \
        --rationale-file ../output/${corpus}_rationale_selection_dev_scibert_mlm.jsonl \
        --label-file ../output/${corpus}_kgat_dev_roberta_large_mlm_pred.jsonl \
        --result-file ../output/${corpus}_merged_predictions.jsonl

    PYTHONPATH=.. python ../pipeline/scifact_eval.py \
        --gold ../../data/claims_dev.jsonl \
        --corpus ../../data/corpus_${corpus}.jsonl \
        --prediction ../output/${corpus}_merged_predictions.jsonl \
        --output ../output/${corpus}_metrics.json
done