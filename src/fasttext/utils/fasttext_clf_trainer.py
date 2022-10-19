import fasttext


class FastTextTrainClf():
    def __init__(self, fasttext, path_to_data: str, epoch: int = 25, lr: float = 0.1, wordngrams: int = 2,
                 minchar: int = 2, maxchar: int = 5):
        self.model = fasttext
        self.path = path_to_data
        self.epoch = epoch
        self.lr = lr
        self.wordNgrams = wordngrams
        self.minchar = minchar
        self.maxchar = maxchar

    def __call__(self, path_to_save_model: str):
        ft_model = self.model.train_supervised(input=self.path, epoch=self.epoch, lr=self.lr,
                                               wordNgrams=self.wordNgrams, minn=self.minchar, maxn=self.maxchar)
        ft_model.save_model(path_to_save_model + '/model_cooking.bin')
