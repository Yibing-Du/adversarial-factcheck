cd kgat
for file in 'baseline' 'advadd_full' 'advadd_min' 'oracle'; do
	python test.py \
		--outdir ../output \
		--test_path ../../data/${file}.json \
		--bert_pretrain ../roberta_large \
		--checkpoint ../checkpoint/kgat_pred_model.best.pt \
		--bert_hidden_dim 1024 \
		--name kgat_${file}.json
	cd ../..
	python eval.py --file kgat/output/kgat_${file}.json
	cd kgat/kgat
done
