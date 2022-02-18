cd roberta_large
wget https://huggingface.co/roberta-large/resolve/main/pytorch_model.bin
wget https://huggingface.co/roberta-large/resolve/main/merges.txt
wget https://huggingface.co/roberta-large/resolve/main/vocab.json

cd checkpoint
sh download.sh

mkdir checkpoint
# Download kgat_pred_model.best.pt: https://drive.google.com/file/d/1RlFUVr_loIL0nJenhrtEAzqeBYfMrJ5S/view?usp=sharing
