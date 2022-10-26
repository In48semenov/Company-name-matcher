# Car number
**Detection of car numbers and their recognition**
# Project structure

Our solution consists of one main parts:
1. **Сlassifier** - solving a classification problem to determine whether the names are one firm

There is a strict interface for each of the parts. Each part is independent of the other. A special interface has been implemented for this module.

This rep presents three methods for solving the problem:
1. Using Bert
2. Using Sentence Transformers 
3. Using FastText

```
.
├── data
├── notebooks               <- Jupyter notebooks
├── README.md               <- The top-level README for developers using this project
├── requirements.txt        <- The requirements file for reproducing the analysis environment
├── weights                 <- Empty folder for saving results
├── src
│   ├── bert                <- Folder that contains bert solution
│   ├── fasttext            <- Folder that contains fasttext solution
│   ├── sentence_bert       <- Folder that contains tentence transformers solution
│   ├── utils
└── tutorial.ipynb          <- Demonstration work
```

# Results 
-
## Metrics
- 
## Performance 
-

# Usage
We tested three different classification models.You can combine them however you like. Be careful with experiments, look at the results.

To demonstrate the results of the project, you can use a [tutorial.ipynb](./tutorial.ipynb) Before using it, you need to install the project dependencies:
```
pip install -r requirements.txt 
```

After installing the dependencies, you need to be in the **root folder** of the repository run commands:
```
# Linux command
chmod +x load_data.sh
./load_data.sh
```

[**Link**](https://drive.google.com/drive/folders/175r-xavYr0N_iv7QhI7fAFiYw1e-qqAh?usp=sharing) to the directory with all weights that are used in this work.
# Reference 
1. [SentenceTransformers](https://www.sbert.net/)
1. [fastText](https://github.com/facebookresearch/fastText)
