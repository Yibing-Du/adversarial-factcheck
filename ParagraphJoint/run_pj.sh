# setup
gdown "https://drive.google.com/u/0/uc?id=1agyrkUGJ0lxTBJpdy1QCyaAAJyxBnoO2"
wget https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin

mkdir output
mkdir output/baseline
mkdir output/advadd

# baseline
# Abstract retrieval

python ComputeBioSentVecAbstractEmbedding.py \
	--claim_file ../data/claims_dev.jsonl \
	--corpus_file ../data/corpus_baseline.jsonl \
	--sentvec_path BioSentVec_PubMed_MIMICIII-bigram_d700.bin \
	--corpus_embedding_pickle ./output/baseline/corpus_paragraph_biosentvec.pkl \
	--claim_embedding_pickle ./output/baseline/claim_biosentvec.pkl

python SentVecAbstractRetriaval.py \
	--claim_file ../data/claims_dev.jsonl \
	--corpus_file ../data/corpus_baseline.jsonl \
	--k_retrieval 3 \
	--claim_retrieved_file ./output/baseline/claim_retrieved.jsonl \
	--scifact_abstract_retrieval_file ./output/baseline/scifact_abstract_retrieval.jsonl \
	--corpus_embedding_pickle ./output/baseline/corpus_paragraph_biosentvec.pkl \
	--claim_embedding_pickle ./output/baseline/claim_biosentvec.pkl

# Rationale selection
python scifact_joint_paragraph_dynamic_prediction.py \
	--corpus_file ../data/corpus_baseline.jsonl \
	--test_file ./output/baseline/claim_retrieved.jsonl \
	--dataset ../data/claims_dev.jsonl \
	--batch_size 25 \
	--k 3 \
	--prediction ./output/baseline/output.jsonl \
	--evaluate \
	--checkpoint scifact_roberta_joint_paragraph_dynamic_fine_tune_ratio=6_lr=5e-6_bert_lr=1e-5_FEVER=5_scifact=12_downsample_good.model

# advadd
# Abstract retrieval
python ComputeBioSentVecAbstractEmbedding.py \
	--claim_file ../data/claims_dev.jsonl \
	--corpus_file ../data/corpus_advadd.jsonl \
	--sentvec_path BioSentVec_PubMed_MIMICIII-bigram_d700.bin \
	--corpus_embedding_pickle ./output/advadd/corpus_paragraph_biosentvec.pkl \
	--claim_embedding_pickle ./output/advadd/claim_biosentvec.pkl

python SentVecAbstractRetriaval.py \
	--claim_file ../data/claims_dev.jsonl \
	--corpus_file ../data/corpus_advadd.jsonl \
	--k_retrieval 3 \
	--claim_retrieved_file ./output/advadd/claim_retrieved.jsonl \
	--scifact_abstract_retrieval_file ./output/advadd/scifact_abstract_retrieval.jsonl \
	--corpus_embedding_pickle ./output/advadd/corpus_paragraph_biosentvec.pkl \
	--claim_embedding_pickle ./output/advadd/claim_biosentvec.pkl

#  Rationale selection
python scifact_joint_paragraph_dynamic_prediction.py \
	--corpus_file ../data/corpus_advadd.jsonl \
	--test_file ./output/advadd/claim_retrieved.jsonl \
	--dataset ../data/claims_dev.jsonl \
	--batch_size 25 \
	--k 3 \
	--prediction ./output/advadd/output.jsonl \
	--evaluate \
	--checkpoint scifact_roberta_joint_paragraph_dynamic_fine_tune_ratio=6_lr=5e-6_bert_lr=1e-5_FEVER=5_scifact=12_downsample_good.model
