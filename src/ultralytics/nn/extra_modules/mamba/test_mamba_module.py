import torch
from mamba_ssm import Mamba

batch, length, dim = 2, 64, 768
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba(
    d_model=dim,
    d_state=16,
    d_conv=4,
    expand=2,
    use_fast_path=False,
).to("cuda")
y = model(x)
assert y.shape == x.shape
