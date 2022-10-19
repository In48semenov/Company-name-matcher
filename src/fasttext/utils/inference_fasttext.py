import fasttext


class FasTextPipeline:
    def __init__(self, path_to_model: str, action: str = 'classification'):
        self.model = fasttext.load_model(path_to_model)
        self.action = action

    def __call__(self, company_1: str, company_2: str):
        if self.action == 'classification':
            self.model.predict(company_1 + " /SEP/ " + company_2)
            return 0 if self.model.predict(company_1 + " /SEP/ " + company_2) == '__label__NEGATIVE' else 1

