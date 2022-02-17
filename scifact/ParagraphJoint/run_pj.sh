for corpus in 'baseline' 'advadd'; do
	# Abstract retrieval
	python ComputeBioSentVecAbstractEmbedding.py \
		--claim_file ../data/claims_dev.jsonl \
		--corpus_file ../data/corpus_${corpus}.jsonl \
		--sentvec_path BioSentVec_PubMed_MIMICIII-bigram_d700.bin \
		--corpus_embedding_pickle ./output/${corpus}/corpus_paragraph_biosentvec.pkl \
		--claim_embedding_pickle ./output/${corpus}/claim_biosentvec.pkl

	python SentVecAbstractRetriaval.py \
		--claim_file ../data/claims_dev.jsonl \
		--corpus_file ../data/corpus_${corpus}.jsonl \
		--k_retrieval 3 \
		--claim_retrieved_file ./output/${corpus}/claim_retrieved.jsonl \
		--scifact_abstract_retrieval_file ./output/${corpus}/scifact_abstract_retrieval.jsonl \
		--corpus_embedding_pickle ./output/${corpus}/corpus_paragraph_biosentvec.pkl \
		--claim_embedding_pickle ./output/${corpus}/claim_biosentvec.pkl

	# Rationale selection
	python scifact_joint_paragraph_dynamic_prediction.py \
		--corpus_file ../data/corpus_${corpus}.jsonl \
		--test_file ./output/${corpus}/claim_retrieved.jsonl \
		--dataset ../data/claims_dev.jsonl \
		--batch_size 25 \
		--k 3 \
		--prediction ./output/${corpus}/output.jsonl \
		--evaluate \
		--checkpoint scifact_roberta_joint_paragraph_dynamic_fine_tune_ratio=6_lr=5e-6_bert_lr=1e-5_FEVER=5_scifact=12_downsample_good.model
done