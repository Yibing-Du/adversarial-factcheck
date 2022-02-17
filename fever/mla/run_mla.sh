gdown "https://drive.google.com/u/1/uc?id=1ePy4gUGVuoquSBZW0GjwAl0FsvP-cS2v"
unzip adversarial_mla_data.zip
rm adversarial_mla_data.zip

cd sentence-selection
sh sentence_selection.sh

cd ../claim-verification-roberta-large
sh claim_verification.sh