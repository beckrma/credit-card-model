import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix

class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Linear(24,12),
            nn.ReLU(),
            nn.Linear(12,6)
        )

        self.decoder = nn.Sequential(
            nn.Linear(6,12),
            nn.ReLU(),
            nn.Linear(12,24),
            nn.ReLU(),
            nn.Linear(24,   input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


if __name__ == "__main__":
    # load data
    df = pd.read_csv("archive/creditcard.csv")

    # choosing a time-based split for making evaluation much more representative of how a deployed system would encounter transactions and handle unseen fraud patterns. 
    # this is important because fraud patterns can change over time, and a model trained on past data may not perform well on future transactions. 
    # by using a time-based split, we can better evaluate the model's ability to generalize to new, unseen data and ensure that it is robust enough to handle evolving fraud patterns in real-world scenarios.
    df = df.sort_values("Time").reset_index(drop=True)

    train_end = int(len(df) * 0.70)
    val_end = int(len(df) * 0.85)

    train_df = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    # geting only normal transactions for training the autoencoder
    normal_train = train_df[train_df['Class'] == 0]
    x_train = normal_train.drop(columns=['Class'])

    x_val = val.drop(columns=['Class'])
    y_val = val['Class']

    x_test = test.drop(columns=['Class'])
    y_test = test['Class']

    # scale data
    scalar = StandardScaler()
    x_train_scaled = scalar.fit_transform(x_train)
    x_val_scaled = scalar.transform(x_val)
    x_test_scaled = scalar.transform(x_test)

    # train autoencoder
    input_dim = x_train_scaled.shape[1]
    model = AutoEncoder(input_dim)

    train_tensor = torch.FloatTensor(x_train_scaled)
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=64,
        shuffle=True
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # training loop

    epochs = 100

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for (batch,) in train_loader:
            optimizer.zero_grad()            
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(
            f"Epoch {epoch+1}: "
            f"{total_loss / len(train_loader):.5f}"
        )

    # validation section
    model.eval()

    x_val_tensor = torch.FloatTensor(x_val_scaled)

    with torch.no_grad():
        val_reconstructed = model(x_val_tensor)

    val_errors = torch.mean(
        (val_reconstructed - x_val_tensor) ** 2,
        dim=1
    )
    
    # choosing thresholds using validation data
    thresholds = [
        torch.quantile(val_errors, 0.99),
        torch.quantile(val_errors, 0.995),
        torch.quantile(val_errors, 0.999)
    ]

    print("\nValidation Results:")
    for threshold in thresholds:
        val_preds = (val_errors > threshold).int().numpy()
        print("Threshold:", threshold.item())
        print(classification_report(y_val, val_preds))
        print(confusion_matrix(y_val, val_preds))


    # final test

    x_test_tensor = torch.FloatTensor(x_test_scaled)

    with torch.no_grad():
        test_reconstructed = model(x_test_tensor)

    test_errors = torch.mean(
        (test_reconstructed - x_test_tensor) ** 2,
        dim=1
    )

    # choosing threshold from validation results
    threshold = 2.883143663406372

    test_preds = (test_errors > threshold).int().numpy()

    print("\nFinal Test Results:")
    print("Threshold:", threshold)
    print(classification_report(y_test, test_preds))
    print(confusion_matrix(y_test, test_preds))

    # save the model
    torch.save(model.state_dict(), "autoencoder_model.pth")