import fasttext
import pandas as pd
import os
from scipy import spatial

module_realpath = os.path.realpath(__file__)
module_folder = os.path.dirname(module_realpath)


class FasTextPipeline:
    def __init__(self, path_to_model: str):
        self.model = fasttext.load_model(path_to_model)

    def __call__(self, company_1: str, company_2: str = None, top_n: int = 10):
        if company_2 is not None:
            self.model.predict(company_1 + " /SEP/ " + company_2)
            return (
                0
                if self.model.predict(company_1 + " /SEP/ " + company_2)
                == "__label__NEGATIVE"
                else 1
            )
        else:
            ft_company_vectors = pd.read_csv(
                f"{module_folder}/../../database/data/ft_company_vectors.csv"
            )
            ft_company_vectors["score"] = ft_company_vectors.apply(
                lambda x: spatial.distance.euclidean(company_1, x["vectors"]), axis=1
            )
            return (
                ft_company_vectors.drop_duplicates(subset=["score"])
                .sort_values(by=["score"])[0:top_n]["senteses"]
                .tolist()
            )
