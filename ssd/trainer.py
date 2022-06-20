import torch
import time
import torch.nn as nn
import pandas as pd

class Trainer:
    def __init__(self, model, dataloader_dict, loss_fn, optimizer, max_epochs):
        self.epoch = 0
        self.model = model
        self.dataloader_dict = dataloader_dict
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.max_epochs = max_epochs

    def train(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('使用デバイス: ', device)

        self.model.to(device)

        torch.backends.cudnn.benchmark = True

        iteration = 1
        epoch_train_loss = 0.0

        epoch_val_loss = 0.0

        logs = []

        for self.epoch in range(self.epoch, self.max_epochs + 1):
            t_epoch_start = time.time()
            t_iter_start = time.time()

            print('----------------------')
            print(f'Epoch {self.epoch+1}/{self.max_epochs}')
            print('----------------------')

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()

                    print(' (train) ')
                else:
                    if ((self.epoch + 1) % 10 == 0):
                        self.model.eval()

                        print('----------------------')
                        print(' (val) ')
                    else:
                        continue

                for images, bboxes, labels in self.dataloader_dict[phase]:
                    images = images.permute(0, 3, 1, 2)
                    images = images.to(device)
                    bboxes = [box.to(device) for box in bboxes]
                    labels = [l.to(device) for l in labels]
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(images)

                        loss_l, loss_c = self.loss_fn(outputs, bboxes, labels)
                        loss = loss_l + loss_c
                        if phase == 'train':
                            loss.backward()

                            nn.utils.clip_grad_value_(
                                self.model.parameters(), clip_value=2.0)

                            self.optimizer.step()

                            if(iteration % 10 == 0):
                                t_iter_finish = time.time()
                                duration = t_iter_finish - t_iter_start
                                print(
                                    f'イテレーション {iteration} || Loss: {loss.item():.4f} || 10iter: {duration:.4f} sec.')
                                t_iter_start = time.time()

                            epoch_train_loss += loss.item()
                            iteration += 1
                        else:
                            epoch_val_loss += loss.item()
            t_epoch_finish = time.time()

            print('----------------------')
            print(
                f'epoch {self.epoch + 1} || Epoch_TRAIN_Loss: {epoch_train_loss:.4f} || Epoch_VAL_Loss: {epoch_val_loss:.4f}')
            print(f'timer: {t_epoch_finish - t_epoch_start:.4f} sec')
            t_epoch_start = time.time()

            log_epoch = {'epoch': self.epoch + 1,
                        'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss}
            logs.append(log_epoch)
            df = pd.DataFrame(logs)
            df.to_csv('log_output.csv')

            epoch_train_loss = 0.0
            epoch_val_loss = 0.0

            if((self.epoch + 1) % 10 == 0):
                torch.save(self.model.state_dict(), 'weights/ssd300_' +
                        str(self.epoch + 1) + '.pth')
