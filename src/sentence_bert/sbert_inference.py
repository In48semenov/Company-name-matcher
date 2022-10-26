import os
import numpy as np
import pandas as pd
from typing import Union
from sentence_transformers import SentenceTransformer, util

module_realpath = os.path.realpath(__file__)
module_folder = os.path.dirname(module_realpath)


class SentBertPipeline:
    threshold = 0.8

    def __init__(self, path_to_model: str, device: str):
        self.device = device
        self.model = SentenceTransformer(path_to_model, device=self.device)

    def __call__(
        self, company_1: str, company_2: str = None, top_n: int = 10
    ) -> Union[int, list]:
        if company_2 is not None:
            embeddings = self.model.encode(
                [company_1, company_2], convert_to_tensor=True, device=self.device
            )
            cosine_score = util.cos_sim(embeddings[0], embeddings[1])[0][0].item()
            if cosine_score > self.threshold:
                return 1
            else:
                return 0
        else:
            embeddings_bd = np.load(
                f"{module_folder}/../../data/database/embeddings_bd.npy"
            )
            companies_bd = pd.read_csv(
                f"{module_folder}/../../data/database/companies_bd.csv"
            )["company"].tolist()
            embeddings = self.model.encode(
                [company_1], convert_to_tensor=True, device=self.device
            )
            cosine_score = (
                util.cos_sim(embeddings, embeddings_bd)[0]
                .cpu()
                .detach()
                .numpy()
                .tolist()
            )

            selected_companies = dict()
            for cmp, cos in zip(companies_bd, cosine_score):
                if cos > self.threshold:
                    selected_companies[cmp] = cos

            if len(selected_companies) > 0:
                return list(
                    dict(
                        sorted(
                            selected_companies.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                    ).keys()
                )[:top_n]
            else:
                return []
