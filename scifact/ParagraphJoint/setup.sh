gdown "https://drive.google.com/u/0/uc?id=1agyrkUGJ0lxTBJpdy1QCyaAAJyxBnoO2"
wget https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin

mkdir output
mkdir output/baseline
mkdir output/advadd

git clone https://github.com/epfml/sent2vec.git
cd sent2vec
make
pip install .
pip install -U numpy