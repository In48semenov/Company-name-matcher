#!/bin/bash

# echo "Bert"
# gdown 1U78Y_oTYQr9Uxa7PFRw7FqKniUnH9pqb -O ./weights/inference/bert.pth

echo "SBert"
gdown "1epf63WPLyDwSNUgKFRoLtuJ5Nyr4Zbjn" -O ./weights/inference/
unzip ./weights/inference/sbert.zip -d ./weights/inference/
rm -rf ./weights/inference/sbert.zip