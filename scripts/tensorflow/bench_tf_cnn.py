import tensorflow as tf
import time

# GPU information
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to allocate only a fraction of GPU memory
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Define a simple CNN
class SimpleCNN(tf.keras.Model):
    def __init__(self):
        super(SimpleCNN, self).__init()
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1))
        self.max_pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Generate random data
batch_size = 64
input_shape = (28, 28, 1)
data = tf.random.normal((batch_size, *input_shape))
labels = tf.random.uniform((batch_size,), minval=0, maxval=10, dtype=tf.int32)

# Create the model
model = SimpleCNN()

# Define loss function and optimizer
criterion = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Number of iterations for benchmarking
num_iterations = 100

# Benchmarking loop
start_time = time.time()
for _ in range(num_iterations):
    # Forward pass
    with tf.GradientTape() as tape:
        outputs = model(data)
        loss = criterion(labels, outputs)

    # Backpropagation and optimization
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

end_time = time.time()

# Calculate the average time per iteration
average_time_per_iteration = (end_time - start_time) / num_iterations

# GPU information
if gpus:
    print(f"GPU Name: {tf.test.gpu_device_name()}")
    print(f"GPU Memory Total: {tf.config.experimental.get_virtual_device_configuration(gpus[0])['memory_limit'] / (1024**3):.2f} GB")

# Print benchmark results
print(f"Average Time per Iteration: {average_time_per_iteration:.6f} seconds")
