cd roberta_large
wget https://huggingface.co/roberta-large/resolve/main/pytorch_model.bin
wget https://huggingface.co/roberta-large/resolve/main/merges.txt
wget https://huggingface.co/roberta-large/resolve/main/vocab.json

cd ../kgat

for file in 'baseline' 'advadd_full' 'advadd_min'; do
	python test.py \
		--outdir ../output \
		--test_path ../../data/${file}.json \
		--bert_pretrain ../roberta_large \
		--checkpoint ../checkpoint/kgat/model.best.pt \
		--bert_hidden_dim 1024 \
		--name kgat_${file}.json
	python ../eval.py --file ../output/kgat_${file}.json
done
