import torch
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
from config import *
import sys
import matplotlib.pyplot as plt
class MethylationDataset(Dataset):
    def __init__(self, series_names, data_folder):
        nsamples =0
        #count the number of samples
        for series_id in series_names:
            series_subfolder = data_folder + "/" + series_id
            pkl_files = [f for f in os.listdir(series_subfolder) if f.endswith(".pkl")]
            nsamples += len(pkl_files)

        X_data = np.zeros((nsamples, NUM_PROBES), dtype=np.float32)
        y_data = np.zeros(nsamples, dtype=np.float32)
        print(f"\n[INFO] Building X_data, y_data with {nsamples} samples, {NUM_PROBES} probes each.")

        i = 0
        for series_id in series_names:
            series_subfolder = data_folder + "/" + series_id
            pkl_files = [f for f in os.listdir(series_subfolder) if f.endswith(".pkl")]
            for pkl_file in pkl_files:
                with open(series_subfolder + "/" + pkl_file, "rb") as f:
                    sample_dict = pickle.load(f)
                    X_data[i,:] = list(sample_dict.values())[:-1]
                    y_data[i] = sample_dict["age"]
                    i += 1
            print(f"Loaded {len(pkl_files)} samples from {series_id}")
            

            
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32)
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AgePredictorMLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AgePredictorMLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x



if __name__ == "__main__":
    dataset = MethylationDataset(SERIES_NAMES, DATA_FOLDER)
    train_size = int(TRAIN_SPLIT_RATIO * len(dataset))
    test_size = len(dataset) - train_size
    torch.manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AgePredictorMLP(dataset.X.shape[1], 256).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.L1Loss()

    print("[INFO] Starting training...")
    test_maes = []
    test_median_errors = []
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            # print(batch_X.shape)
            # print(batch_y.shape)
            # sys.exit()
            optimizer.zero_grad()
            loss = criterion(model(batch_X).squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_mae = total_loss / len(train_loader)
        scheduler.step()

        model.eval()
        total_test_loss = 0
        all_errors = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                preds = model(batch_X).squeeze()
                total_test_loss += criterion(preds, batch_y).item()
                all_errors.extend((preds - batch_y).abs().cpu().numpy())
        test_mae = total_test_loss / len(test_loader)
        median_error = float(np.median(all_errors))
        test_maes.append(test_mae)
        test_median_errors.append(median_error)

        print(f"[Epoch {epoch+1:02d}/{NUM_EPOCHS}] "
              f"Train MAE: {train_mae:.4f}, "
              f"Test MAE: {test_mae:.4f}, "
              f"Test Median Error: {median_error:.4f}")

    torch.save(model, "age_predictor_mlp.pth")
    print("[INFO] Model saved to age_predictor_mlp.pth")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), test_maes, label='Test MAE')
    plt.plot(range(1, NUM_EPOCHS + 1), test_median_errors, label='Test Median Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Test MAE and Median Error over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_metrics.png")
    plt.show()