import torch
import torch.nn as nn
import numpy as np
import os

from UNet import UNet

def train(device, model, train_loader, val_loader,
          epochs=200, lr=1e-3,
          save_path="saved_models"
          ):

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    
    for epoch in epochs:

        train_loss_epoch = []
        val_loss_epoch = []

        model.train()

        for i, batch in enumerate(train_loader):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

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

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            preds = model(inputs)

            loss = criterion(preds, labels)

            test_losses.append(loss.item())
        
    test_loss = np.array(test_losses).mean()
    print(f"Test loss: {test_loss}")


def get_psnr(clean, denoised, max_pixel_value=1.0):
    """Computes PSNR (Peak Signal-to-Noise Ratio) between clean and denoised images."""
    mse = F.mse_loss(denoised, clean, reduction='mean')  # Mean Squared Error
    psnr = 10 * torch.log10(max_pixel_value**2 / mse)  # PSNR formula
    return psnr.item()



def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    train_loader = get_data_loader()
    val_loader = get_data_loader()
    test_loader = get_data_loader()

    unet = UNet(in_channels=?,
                 out_channels=?,
                 init_channels=8)
    
    unet = train(device=device, model=unet,
                  train_loader=train_loader,
                  val_loader=val_loader,
                  epochs=200,
                  lr=1e-3)
    
    
    test(device=device, model=unet, test_loader=test_loader)


if __name__ == "__main__":
    main()