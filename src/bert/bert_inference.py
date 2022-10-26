import os
import torch
import pandas as pd
import transformers
from tqdm import tqdm

module_realpath = os.path.realpath(__file__)
module_folder = os.path.dirname(module_realpath)


class BertPipeline:
    threshold = 0.824992835521698

    def __init__(self, tokenizer: transformers, path_to_model: str, device: str):
        self.model = torch.load(path_to_model, map_location=device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    def _get_tokens(self, cmp_1: str, smp_2: str):
        companies = "[CLS] " + cmp_1 + " [SEP] " + smp_2 + " [SEP]"
        tokens = self.tokenizer(
            companies,
            truncation=True,
            add_special_tokens=False,
            max_length=120,
            pad_to_max_length=True,
            return_tensors="pt",
        )

        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }

    @torch.no_grad()
    def __call__(self, company_1: str, company_2: str = None, top_n: int = 10):
        if company_2 is not None:
            tokens = self._get_tokens(company_1, company_2)
            logits = self.model(
                tokens["input_ids"].to(self.device),
                attention_mask=tokens["attention_mask"].to(self.device),
            ).logits

            proba_logit = torch.nn.Softmax(dim=1)(logits.cpu())[0][1].item()

            if proba_logit > self.threshold:
                return 1
            else:
                return 0
        else:
            companies_bd = pd.read_csv(f"{module_folder}/../../data/database/companies_bd.csv")

            selected_companies = dict()
            for idx, company in tqdm(enumerate(companies_bd["company_preprocess"])):
                tokens = self._get_tokens(company_1, company)
                logits = self.model(
                    tokens["input_ids"].to(self.device),
                    attention_mask=tokens["attention_mask"].to(self.device),
                ).logits
                proba_logit = torch.nn.Softmax(dim=1)(logits.cpu())[0][1].item()
                if proba_logit > self.threshold:
                    selected_companies[companies_bd["company"].iloc[idx]] = proba_logit

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
