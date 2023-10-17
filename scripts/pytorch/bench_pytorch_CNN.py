import torch
import torch.nn as nn
import torch.optim as optim
import time

# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32 * 6 * 6, 10)  # Assuming input images are 32x32

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(-1, 32 * 6 * 6)
        x = self.fc1(x)
        return x

# Generate random data (assuming 3-channel images of size 32x32)
batch_size = 64
input_channels = 3
input_size = 32
data = torch.randn(batch_size, input_channels, input_size, input_size).cuda()
labels = torch.randint(0, 10, (batch_size,)).cuda()

# Create the model and move it to GPU
model = SimpleCNN().cuda()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Number of iterations for benchmarking
num_iterations = 100

# Benchmarking loop
start_time = time.time()
for _ in range(num_iterations):
    # Forward pass
    outputs = model(data)

    # Compute loss
    loss = criterion(outputs, labels)

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

end_time = time.time()

# Calculate the average time per iteration
average_time_per_iteration = (end_time - start_time) / num_iterations

# Print benchmark results
print(f"Average time per iteration: {average_time_per_iteration:.6f} seconds")

# Print GPU information
print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
