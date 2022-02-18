# setup
# download CorefRoBERTa-large from https://drive.google.com/drive/folders/1Ls3AEGGD4Z7Wgx73woGCwnBt23CxGioc
# download corefbert_model from https://drive.google.com/file/d/1yX68xv4gbAgEh8zXruBj7Qyg_78xCW1X/view?usp=sharing

for file in 'baseline' 'advadd_full' 'advadd_min' 'oracle'; do
	python FEVER/test.py \
		--model_type roberta \
		--outdir output/ \
		--test_path ../data/${file}.json \
		--bert_pretrain CorefRoBERTa-large \
		--batch_size 32 \
		--checkpoint model/model.best.pt \
		--name corefbert_${file}.json
	cd ..
    python eval.py --file corefbert/output/corefbert_${file}.json
    cd corefbert
done