cd roberta_large
wget https://huggingface.co/roberta-large/resolve/main/pytorch_model.bin
wget https://huggingface.co/roberta-large/resolve/main/merges.txt
wget https://huggingface.co/roberta-large/resolve/main/vocab.json

cd checkpoint
sh download.sh