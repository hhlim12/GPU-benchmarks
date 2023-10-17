import tensorflow as tf
import time

# Define a simple neural network using the Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(512,)),
    tf.keras.layers.Dense(10)
])

# Generate random data
batch_size = 64
input_size = 512
data = tf.random.normal((batch_size, input_size))
labels = tf.random.uniform((batch_size,), maxval=10, dtype=tf.int32)

# Define loss function and optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# Number of iterations for benchmarking
num_iterations = 100

# Benchmarking loop
start_time = time.time()
for _ in range(num_iterations):
    with tf.GradientTape() as tape:
        logits = model(data, training=True)
        loss_value = loss_fn(labels, logits)
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
end_time = time.time()

# Calculate the average time per iteration
average_time_per_iteration = (end_time - start_time) / num_iterations

# Print benchmark results
print(f"Average time per iteration: {average_time_per_iteration:.6f} seconds")

# Print GPU information (if using a GPU)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"GPU Device Name: {gpus[0].name}")
    print(f"GPU Memory Total: {tf.config.experimental.get_memory_info(gpus[0]).total / (1024**3):.2f} GB")
