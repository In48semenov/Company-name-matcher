from typing import Union
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


class SentBertPipeline:
    threshold = 0.8

    def __init__(self, path_to_model: str = '../../weights/sbert', device: str = 'cpu'):
        self.device = device
        self.model = SentenceTransformer(path_to_model, device=self.device)

    def __call__(self, company_1: str, company_2: str = None, top_n: int = 10) -> Union[int, list]:
        if company_2 is not None:
            embeddings = self.model.encode(
                [
                    company_1, company_2
                ],
                convert_to_tensor=True, device=self.device
            )
            cosine_score = util.cos_sim(embeddings[0], embeddings[1])[0][0].item()
            if cosine_score > self.threshold:
                return 1
            else:
                return 0
        else:
            embeddings_bd = np.load('../data/embeddings_bd.npy')
            companies_bd = pd.read_csv('../data/companies_bd.csv')['company'].tolist()
            embeddings = self.model.encode(
                [
                    company_1
                ],
                convert_to_tensor=True, device=self.device
            )
            cosine_score = util.cos_sim(embeddings, embeddings_bd)[0].cpu().detach().numpy().tolist()

            selected_companies = dict()
            for cmp, cos in zip(companies_bd, cosine_score):
                if cos > self.threshold:
                    selected_companies[cmp] = cos

            if len(selected_companies) > 0:
                return list(
                    dict(
                        sorted(selected_companies.items(), key=lambda item: item[1], reverse=True)
                    ).keys()
                )[:top_n]
            else:
                return []
