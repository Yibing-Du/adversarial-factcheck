for corpus in 'baseline' 'advadd'; do
    #abstract retrieval
    python ../abstract_retrieval/tfidf.py \
        --corpus ../../data/corpus_${corpus}.jsonl \
        --dataset ../../data/claims_dev.jsonl \
        --k 100 \
        --min-gram 1 \
        --max-gram 2 \
        --output ../output/${corpus}_abstract_retrieval_dev_top100.jsonl

    # abstract reranking
    python ../abstract_rerank/inference.py \
            -checkpoint ../model/abstract_scibert_mlm/pytorch_model.bin \
            -corpus ../../data/corpus_${corpus}.jsonl \
            -abstract_retrieval ../output/${corpus}_abstract_retrieval_dev_top100.jsonl \
            -dataset ../../data/claims_dev.jsonl \
            -outpath ../output/${corpus}_abstract_rerank_dev_mlm.jsonl \
            -max_query_len 32 \
            -max_seq_len 256 \
            -batch_size 32

    # rationale selection
    python ../rationale_selection/transformer.py \
        --corpus ../../data/corpus_${corpus}.jsonl \
        --dataset ../../data/claims_dev.jsonl \
        --abstract-retrieval ../output/${corpus}_abstract_rerank_dev_mlm.jsonl \
        --model ../model/rationale_scibert_mlm/ \
        --output-flex ../output/${corpus}_rationale_selection_dev_scibert_mlm.jsonl

    # label prediction
    python ../kgat/test.py \
        --outdir ../output \
        --corpus ../../data/corpus_${corpus}.jsonl \
        --evidence_retrieval ../output/${corpus}_rationale_selection_dev_scibert_mlm.jsonl \
        --dataset ../../data/claims_dev.jsonl \
        --checkpoint ../model/kgat_roberta_large_mlm/model.best.pt \
        --pretrain ../mlm_model/roberta_large_mlm \
        --name ${corpus}_kgat_dev_roberta_large_mlm \
        --roberta \
        --bert_hidden_dim 1024
done
