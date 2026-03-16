import torch
import torch.nn as nn

# 1. Create a tensor
x = torch.tensor([1.0, 2.0, 3.0])
print("Tensor:", x)
print("Shape:", x.shape)
print("dtype:", x.dtype)

# 2. Autograd
w = torch.tensor(2.0, requires_grad=True)
y = w * x
loss = y.sum()
loss.backward()
print("Gradient of w:", w.grad)

# 3. A tiny nn.Module
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

model = TinyModel()
print("\nModel parameters:")
for name, param in model.named_parameters():
    print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")

# 4. One full training step
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

x_batch = torch.randn(8, 3)
y_batch = torch.randn(8, 1)

print("\n--- Training step ---")
print("Loss BEFORE:", criterion(model(x_batch), y_batch).item())

optimizer.zero_grad()
predictions = model(x_batch)
loss = criterion(predictions, y_batch)
loss.backward()
optimizer.step()

print("Loss AFTER: ", criterion(model(x_batch), y_batch).item())
print("--- Done ---")

# 5. The LSTM shape
lstm = nn.LSTM(input_size=21, hidden_size=128, batch_first=True)

fake_input = torch.randn(32, 7, 21)
out, (h_n, c_n) = lstm(fake_input)

print(f"\nLSTM input:  {fake_input.shape}")
print(f"LSTM output: {out.shape}")
print(f"Last step:   {out[:, -1, :].shape}")
print(f"h_n:         {h_n.shape}")