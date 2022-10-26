!/bin/bash

echo "Bert"
gdown 1U78Y_oTYQr9Uxa7PFRw7FqKniUnH9pqb -O ./weights/inference/bert.pth

echo "SBert"
gdown "1epf63WPLyDwSNUgKFRoLtuJ5Nyr4Zbjn" -O ./weights/inference/
unzip ./weights/inference/sbert.zip -d ./weights/inference/
rm -rf ./weights/inference/sbert.zip

echo "FastText"
gdown "10ViWC49SRpXheS6od23QE8yrzGe33i2T" -O ./weights/inference/
unzip ./weights/inference/fasttext_model.zip -d ./weights/inference/fasttext
rm -rf ./weights/inference/fasttext_model.zip
