import torch
from nflows.distributions.normal import StandardNormal
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.flows import Flow

# Normalizing Flowの設定
def create_flow(input_dim, num_layers=5, hidden_dim=64):
    # Transformのリスト
    transforms = []
    for _ in range(num_layers):
        transforms.append(MaskedAffineAutoregressiveTransform(features=input_dim, hidden_features=hidden_dim))
    # Base distribution
    base_distribution = StandardNormal([input_dim])
    # Flow
    return Flow(transform=torch.nn.Sequential(*transforms), distribution=base_distribution)

# モデルの作成
input_dim = encoder_features.shape[1]
flow_model = create_flow(input_dim=input_dim)