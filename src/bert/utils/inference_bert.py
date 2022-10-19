import torch
import transformers


class BertPipeline:

    def __init__(self, path_to_model: str, tokenizer: transformers, device: str = 'cpu', debug: bool = False):
        self.model = torch.load(path_to_model, map_location=device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.debug= debug

    def _get_tokens(self, cmp_1: str, smp_2: str):
        companies = '[CLS] ' + cmp_1 + ' [SEP] ' + smp_2 + ' [SEP]'
        tokens = self.tokenizer(
            companies,
            truncation=True,
            add_special_tokens=False,
            max_length=60,
            pad_to_max_length=True,
            return_tensors='pt'
        )

        return {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask']
        }

    @torch.no_grad()
    def __call__(self, company_1: str, company_2: str):
        tokens = self._get_tokens(company_1, company_2)
        logits = self.model(
            tokens['input_ids'].to(self.device), attention_mask=tokens['attention_mask'].to(self.device)
        ).logits
        if self.debug:
            cls = torch.nn.Softmax(dim=1)(logits.cpu())[0][1].item()
        else:
            cls = torch.argmax(logits.cpu(), dim=1).item()
        return cls
