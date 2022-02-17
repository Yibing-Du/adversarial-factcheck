gdown "https://drive.google.com/uc?id=1OOyFxO0NkHrNp7Us3gQUsjjpbzFPYul6"
mkdir covidfact-roberta
mv checkpoint_best.pt covidfact-roberta
python result.py
