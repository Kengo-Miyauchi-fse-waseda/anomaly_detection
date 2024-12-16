from modeling.Flow_based.RealNVP import RealNVP
# flow based model
def creata_NF():
# Normalizing Flowの学習
    nf_model = RealNVP(dims=[features_train.shape[1]], cfg={"layers": 6})
    nf_model.to(device)
    train_dataloader = create_dataloader(features_train.numpy(), batch_size=256, need_shuffle=True)
    test_dataloader = create_dataloader(features_test.numpy(), batch_size=256, need_shuffle=False)

    optimizer = optim.Adam(nf_model.parameters(), lr=1e-3)

    for epoch in range(50):
        nf_model.train()
        epoch_loss = 0.0
        for batch in train_dataloader:
            batch = batch.to(device)
            z, log_jacobian = nf_model.forward(batch)
            loss = torch.mean(-nf_model.base_distribution.log_prob(z) - log_jacobian)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}")

# 1. 正常データの異常スコアを計算
# 学習データを使って異常スコアを計算（正常データのみ）
normal_scores = []
nf_model.eval()
with torch.no_grad():
    for batch in train_dataloader:
        batch = batch.to(device)
        z, log_jacobian = nf_model.forward(batch)
        log_likelihood = nf_model.base_distribution.log_prob(z) + log_jacobian
        normal_scores.extend(-log_likelihood.cpu().numpy())

# 2. 閾値の決定（例: パーセンタイル法）
# 正常スコアの上位5%を閾値に設定
threshold = np.percentile(normal_scores, 95)  # 上位5%に該当する値

# 3. 異常スコアを計算（テストデータ）
test_scores = []
with torch.no_grad():
    for batch in test_dataloader:
        batch = batch.to(device)
        z, log_jacobian = nf_model.forward(batch)
        log_likelihood = nf_model.base_distribution.log_prob(z) + log_jacobian
        test_scores.extend(-log_likelihood.cpu().numpy())