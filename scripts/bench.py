import torch
import torch.nn as nn
import torch.optim as optim
import time

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Generate random data
batch_size = 64
input_size = 512
data = torch.randn(batch_size, input_size).cuda()
labels = torch.randint(0, 10, (batch_size,)).cuda()

# Create the model and move it to GPU
model = SimpleNet().cuda()

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
