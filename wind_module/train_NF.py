# 特徴量変換と同時にFlowの学習もした方がよさそう


import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

# データセットを作成
dataset = TensorDataset(encoder_features)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Optimizerの設定
optimizer = Adam(flow_model.parameters(), lr=1e-3)

# 学習ループ
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        batch = batch[0].to(torch.float32)  # バッチを取得
        
        # 負の対数尤度を計算
        loss = -flow_model.log_prob(inputs=batch).mean()
        
        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")
