from sklearn.metrics import f1_score, classification_report
import torch
import transformers
from tqdm import tqdm


class BertTrainClf:

    def __init__(self, model: transformers, trainDataloader: torch, valDataloader: torch, criteriation: torch,
                 optimizer: torch, scheduler: torch = None, device: str = 'cuda:0',
                 model_name: str = 'BertNameCompany'):

        self.model = model
        self.trainDataloader = trainDataloader
        self.valDataloader = valDataloader
        self.criteriation = criteriation
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model_name = model_name

        self.model.to(self.device)

    def _train(self):
        print('Training')

        self.model.train()
        train_loss_list = []
        true_label, predict_label = [], []
        prog_bar = tqdm(self.trainDataloader, total=len(self.trainDataloader))

        for i, data in enumerate(prog_bar):
            self.optimizer.zero_grad()

            input_ids = torch.squeeze(data['input_ids']).to(self.device)
            attention_mask = torch.squeeze(data['attention_mask']).to(self.device)
            labels = torch.squeeze(data['label'])

            logits = self.model(input_ids, attention_mask=attention_mask).logits

            curr_loss = self.criteriation(logits.cpu(), labels.cpu())

            train_loss_list.append(curr_loss.item())
            true_label.extend(labels)
            predict_label.extend(torch.argmax(logits.cpu(), dim=1))

            # optimization
            curr_loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            prog_bar.set_description(desc=f"Loss: {curr_loss.item():.4f}")

        train_loss = sum(train_loss_list) / (i + 1)
        f1 = f1_score(true_label, predict_label, average='macro')

        return train_loss, f1

    @torch.no_grad()
    def _evolution(self):
        print('Validating')

        self.model.eval()
        val_loss_list = []
        true_label, predict_label = [], []
        prog_bar = tqdm(self.valDataloader, total=len(self.valDataloader))

        for i, data in enumerate(prog_bar):
            input_ids = torch.squeeze(data['input_ids']).to(self.device)
            attention_mask = torch.squeeze(data['attention_mask']).to(self.device)
            labels = torch.squeeze(data['label'])

            logits = self.model(input_ids, attention_mask=attention_mask).logits

            curr_loss = self.criteriation(logits.cpu(), labels)

            val_loss_list.append(curr_loss.item())
            true_label.extend(labels)
            predict_label.extend(torch.argmax(logits.cpu(), dim=1))

            prog_bar.set_description(desc=f"Loss: {curr_loss.item():.4f}")

        val_loss = sum(val_loss_list) / (i + 1)
        f1 = f1_score(true_label, predict_label, average='macro')
        return val_loss, f1

    def __call__(self, num_epochs):

        train_loss_history, val_loss_history = [], []
        train_f1_history, val_f1_history = [], []

        for epoch in range(num_epochs):
            print(f"\nEPOCH {epoch + 1} of {num_epochs}")

            train_loss, train_f1 = self._train()
            val_loss, val_f1 = self._evolution()

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            train_f1_history.append(train_f1)
            val_f1_history.append(val_f1)

            print(f'f1_macro_train: {train_f1:.3f}\nf1_macro_val: {val_f1:.3f}')
            if epoch == 0:
                best_loss = val_loss
                best_f1 = val_f1

            if (val_loss <= best_loss) and (epoch != 0) and (best_f1 <= val_f1):
                best_loss = val_loss
                best_f1 = val_f1
                try:
                    torch.save(self.model, f'../weights/{self.model_name}_best.pth')
                    print('Save best model.')
                except:
                    print("Can't save best model!")

            try:
                torch.save(self.model, f'../weights/{self.model_name}_last.pth')
            except:
                print("Can't save last model!")

        return {
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
            'train_f1_history': train_f1_history,
            'val_f1_history': val_f1_history
        }
