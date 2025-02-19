import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional as F

from UNet import UNet
from dataset_class import SpeechTestDataset, SpeechTrainDataset
from evaluation import get_psnr

def train(device, model, train_loader, val_loader,
          epochs=200, lr=1e-3,
          save_path="saved_models"
          ):

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    
    for epoch in range(epochs):

        train_loss_epoch = []
        val_loss_epoch = []

        model.train()

        for i, batch in enumerate(train_loader):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.unsqueeze(1), labels.unsqueeze(1)

            optimizer.zero_grad()

            preds = model(inputs)

            loss = criterion(preds, labels)
            loss.backward()

            optimizer.step()

            train_loss_epoch.append(loss.item())
        
        model.eval()

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                inputs, labels = inputs.unsqueeze(1), labels.unsqueeze(1)
                preds = model(inputs)

                loss = criterion(preds, labels)

                val_loss_epoch.append(loss.item())

        train_loss = np.array(train_loss_epoch).mean()
        val_loss = np.array(val_loss_epoch).mean()
        print('\n', f" *** Epoch {epoch:03d} ***\n Train loss: {train_loss:.3f}\n Validation loss: {val_loss:.3f}")
            
    print("Training finished.")


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save(model.state_dict(), save_path) # save trained model

    return model

def test(device, model, test_loader):

    criterion = nn.MSELoss()

    test_losses = []
    test_psnrs = []

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            preds = model(inputs)

            loss = criterion(preds, labels)

            test_losses.append(loss.item())
            test_psnrs.append(get_psnr(preds.cpu(), labels.cpu()))
        
    test_loss = np.array(test_losses).mean()
    test_psnr = np.array(test_psnrs).mean()
    print(f"Test loss: {test_loss}")
    print(f"Test PSNR: {test_psnr}")


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    train_dataset = SpeechTrainDataset(root_dir='.')

    # split the dataset into train and validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, shuffle=False)
    val_loader = DataLoader(val_dataset, shuffle=False)

    test_dataset = SpeechTestDataset(root_dir='.')
    test_loader = DataLoader(test_dataset, shuffle=False)

    unet = UNet(in_channels=1,
                 out_channels=1,
                 init_channels=8)
    unet.to(device)
    
    unet = train(device=device, model=unet,
                  train_loader=train_loader,
                  val_loader=val_loader,
                  epochs=200,
                  lr=1e-3)
    
    
    test(device=device, model=unet, test_loader=test_loader)


if __name__ == "__main__":
    main()