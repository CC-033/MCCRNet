import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score
import pandas as pd  
import os
from model import MCCRNet

class Solver(object):
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True, model=None, save_name='Proposed'):
        self.hp = hyp_params
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

        self.is_train = is_train
        self.model = model if model is not None else MCCRNet(hyp_params)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.best_val_acc = 0.0  
        self.patience = hyp_params.patience

        class_counts = [999, 2221, 1826]  
        total_samples = sum(class_counts)
        class_weights = [total_samples / c for c in class_counts] 
        weight_tensor = torch.tensor(class_weights).to(self.device)

        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor, reduction="mean")

       
        if self.is_train:
            self.init_model_parameters()

        self.output_csv_path = os.path.join(f"../src/outputs/", f"{save_name}.csv")
        self.best_model_path = os.path.join(f"../src/outputs/", f"{save_name}.pth")

        if not os.path.exists(self.output_csv_path):
            df = pd.DataFrame(columns=["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc"])
            df.to_csv(self.output_csv_path, index=False)

    def train_and_eval(self):
        print("Starting training and evaluation.")
        for epoch in range(1, self.hp.num_epochs + 1):
            print(f"Epoch {epoch} / {self.hp.num_epochs}")

            # Training
            train_loss, train_acc = self.train(epoch)
            print(f"Training loss: {train_loss}, Training accuracy: {train_acc}")

            # Validation
            val_loss, val_acc = self.evaluate(test=False)
            print(f"Validation loss: {val_loss}, Validation accuracy: {val_acc}")
            self.scheduler_main.step(val_loss) 

            # Save results to CSV
            self.save_to_csv(epoch, train_loss, train_acc, val_loss, val_acc)

       
            # if val_acc > self.best_val_acc:
            #     self.best_val_acc = val_acc
            #     self.save_model(self.best_model_path)
            #
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model(self.best_model_path)
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if self.early_stop_counter >= self.patience:
                print("Early stopping triggered.")
                break  

    def train(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for i_batch, batch_data in enumerate(self.train_loader):
            frames, poses, audio, is09, y = batch_data 
            frames = frames.to(self.device)
            poses = poses.to(self.device)
            audio = audio.to(self.device)
            is09 = is09.to(self.device)  

            self.optimizer_main.zero_grad()

            outputs = self.model(frames, poses, audio, is09)  
            y = y.to(outputs.device)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer_main.step()

            total_loss += loss.item()


            preds = torch.argmax(outputs, dim=1)
            total_correct += torch.sum(preds == y).item()
            total_samples += y.size(0)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def evaluate(self, test=False):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        results, truths = [], []
        loader = self.test_loader if test else self.dev_loader

        with torch.no_grad():
            for batch_data in loader:
                frames, poses, audio, is09, y = batch_data 
                frames = frames.to(self.device)
                poses = poses.to(self.device)
                audio = audio.to(self.device)
                is09 = is09.to(self.device)  
                y = y.to(self.device)

                outputs = self.model(frames, poses, audio, is09) 
                y = y.to(outputs.device)
                loss = self.criterion(outputs, y)
                total_loss += loss.item()

     
                preds = torch.argmax(outputs, dim=1)
                total_correct += torch.sum(preds == y).item()
                total_samples += y.size(0)

              
                results.append(outputs)
                truths.append(y)

        avg_loss = total_loss / len(loader)
        accuracy = total_correct / total_samples

       
        results = torch.cat(results)
        truths = torch.cat(truths)

        return avg_loss, accuracy  

    def save_to_csv(self, epoch, train_loss, train_acc, val_loss, val_acc):
       
        df = pd.DataFrame({
            "Epoch": [epoch],
            "Train Loss": [train_loss],
            "Train Acc": [train_acc],
            "Val Loss": [val_loss],
            "Val Acc": [val_acc]
        })
        df.to_csv(self.output_csv_path, mode='a', header=False, index=False)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved at {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
