# setup
pip install -r requirements.txt

sh download-model.sh rationale roberta_large scifact
sh download-model.sh label roberta_large fever_scifact

mkdir output
mkdir output/baseline
mkdir output/advadd
