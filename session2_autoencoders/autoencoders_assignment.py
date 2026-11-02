"""
SESSION 2: Autoencoders & Variational Autoencoders (VAE)
=======================================================
1. Simple Autoencoder (MSE loss, 5 epochs) - Original vs Reconstruction side by side
2. Comparison of Loss Functions: MSE vs Binary Crossentropy (BCE) for 3 test images
3. Image Denoising: Add random Gaussian/uniform noise to 10 test images and reconstruct
4. Variational Autoencoder (VAE): Latent space sampling (5 random vectors) to generate new items
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

OUTPUT_DIR = "session2_autoencoders/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("LOADING FASHION MNIST DATASET...")
print("=" * 70)

(x_train, _), (x_test, _) = keras.datasets.fashion_mnist.load_data()

# Normalize pixel values to [0, 1] and reshape to (N, 28, 28, 1) / flat (N, 784)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Use a representative subset for fast CPU execution
train_subset = x_train[:25000]
test_subset = x_test[:1000]

# Flattened versions for dense autoencoder
x_train_flat = train_subset.reshape((len(train_subset), 784))
x_test_flat = test_subset.reshape((len(test_subset), 784))

print(f"Train subset shape: {x_train_flat.shape}, Test subset shape: {x_test_flat.shape}")


# ==============================================================================
# 1. SIMPLE AUTOENCODER (MSE LOSS, 5 EPOCHS)
# ==============================================================================
print("\n" + "=" * 70)
print("TASK 1: TRAINING SIMPLE AUTOENCODER (MSE LOSS, 5 EPOCHS)")
print("=" * 70)

def build_simple_autoencoder():
    # Encoder
    input_img = keras.Input(shape=(784,))
    encoded = layers.Dense(128, activation='relu')(input_img)
    encoded = layers.Dense(64, activation='relu')(encoded)
    bottleneck = layers.Dense(32, activation='relu')(encoded) # Compressed representation
    
    # Decoder
    decoded = layers.Dense(64, activation='relu')(bottleneck)
    decoded = layers.Dense(128, activation='relu')(decoded)
    output_img = layers.Dense(784, activation='sigmoid')(decoded)
    
    autoencoder = Model(input_img, output_img, name="Autoencoder_MSE")
    return autoencoder

ae_mse = build_simple_autoencoder()
ae_mse.compile(optimizer='adam', loss='mean_squared_error')
ae_mse.fit(
    x_train_flat, x_train_flat,
    epochs=5,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test_flat, x_test_flat),
    verbose=1
)

# Reconstruct a sample image
sample_idx = 0
orig_img = x_test_flat[sample_idx:sample_idx+1]
recon_img = ae_mse.predict(orig_img)

# Save Side-by-Side comparison
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].imshow(orig_img.reshape(28, 28), cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(recon_img.reshape(28, 28), cmap='gray')
axes[1].set_title("Reconstructed (MSE)")
axes[1].axis('off')

plt.suptitle("Task 1: Autoencoder Reconstruction (5 Epochs, MSE Loss)")
plt.tight_layout()
task1_path = os.path.join(OUTPUT_DIR, "reconstruction_mse.png")
plt.savefig(task1_path, dpi=150)
plt.close()
print(f"Task 1 result saved to '{task1_path}'")


# ==============================================================================
# 2. BINARY CROSSENTROPY (BCE) VS MEAN SQUARED ERROR (MSE)
# ==============================================================================
print("\n" + "=" * 70)
print("TASK 2: TRAINING AUTOENCODER WITH BINARY CROSSENTROPY (BCE) LOSS")
print("=" * 70)

ae_bce = build_simple_autoencoder()
ae_bce.compile(optimizer='adam', loss='binary_crossentropy')
ae_bce.fit(
    x_train_flat, x_train_flat,
    epochs=5,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test_flat, x_test_flat),
    verbose=1
)

# Compare on 3 test images visually
indices = [3, 7, 12] # Distinct fashion items (e.g. boot, shirt, sneaker)
test_samples = x_test_flat[indices]
recon_mse_samples = ae_mse.predict(test_samples)
recon_bce_samples = ae_bce.predict(test_samples)

fig, axes = plt.subplots(3, 3, figsize=(9, 9))
labels = ["Image 1 (Sneaker/Shoe)", "Image 2 (Shirt)", "Image 3 (Bag/Boot)"]

for i in range(3):
    # Column 1: Original
    axes[i, 0].imshow(test_samples[i].reshape(28, 28), cmap='gray')
    axes[i, 0].set_title(f"Original {i+1}")
    axes[i, 0].axis('off')
    
    # Column 2: MSE Reconstruction
    axes[i, 1].imshow(recon_mse_samples[i].reshape(28, 28), cmap='gray')
    axes[i, 1].set_title(f"MSE Loss Recon")
    axes[i, 1].axis('off')
    
    # Column 3: BCE Reconstruction
    axes[i, 2].imshow(recon_bce_samples[i].reshape(28, 28), cmap='gray')
    axes[i, 2].set_title(f"BCE Loss Recon (Sharper)")
    axes[i, 2].axis('off')

plt.suptitle("Task 2: Reconstruction Quality Comparison: MSE vs BCE Loss (3 Test Images)", fontsize=13)
plt.tight_layout()
task2_path = os.path.join(OUTPUT_DIR, "comparison_mse_vs_bce.png")
plt.savefig(task2_path, dpi=150)
plt.close()
print(f"Task 2 result saved to '{task2_path}'")


# ==============================================================================
# 3. DENOISING AUTOENCODER
# ==============================================================================
print("\n" + "=" * 70)
print("TASK 3: IMAGE DENOISING WITH AUTOENCODER (10 TEST IMAGES)")
print("=" * 70)

# Add random noise factor
noise_factor = 0.35
x_train_noisy = x_train_flat + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_flat.shape)
x_test_noisy = x_test_flat + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_flat.shape)

x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)

# Denoising Autoencoder
denoiser = build_simple_autoencoder()
denoiser.compile(optimizer='adam', loss='binary_crossentropy')
denoiser.fit(
    x_train_noisy, x_train_flat, # Map noisy -> clean
    epochs=5,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test_noisy, x_test_flat),
    verbose=1
)

# Select 10 test images
ten_orig = x_test_flat[:10]
ten_noisy = x_test_noisy[:10]
ten_denoised = denoiser.predict(ten_noisy)

fig, axes = plt.subplots(3, 10, figsize=(20, 6))
for j in range(10):
    # Row 1: Original
    axes[0, j].imshow(ten_orig[j].reshape(28, 28), cmap='gray')
    axes[0, j].axis('off')
    if j == 0:
        axes[0, j].set_ylabel("Original", fontsize=12)
    
    # Row 2: Noisy
    axes[1, j].imshow(ten_noisy[j].reshape(28, 28), cmap='gray')
    axes[1, j].axis('off')
    if j == 0:
        axes[1, j].set_ylabel("Noisy Input", fontsize=12)
        
    # Row 3: Denoised
    axes[2, j].imshow(ten_denoised[j].reshape(28, 28), cmap='gray')
    axes[2, j].axis('off')
    if j == 0:
        axes[2, j].set_ylabel("Denoised Output", fontsize=12)

plt.suptitle("Task 3: Denoising Autoencoder Results on 10 Fashion MNIST Images (Original vs Noisy vs Denoised)", fontsize=14)
plt.tight_layout()
task3_path = os.path.join(OUTPUT_DIR, "denoising_results.png")
plt.savefig(task3_path, dpi=150)
plt.close()
print(f"Task 3 result saved to '{task3_path}'")


# ==============================================================================
# 4. VARIATIONAL AUTOENCODER (VAE) & LATENT SPACE SAMPLING
# ==============================================================================
print("\n" + "=" * 70)
print("TASK 4: BUILDING & TRAINING VARIATIONAL AUTOENCODER (VAE)")
print("=" * 70)

latent_dim = 8

class Sampling(layers.Layer):
    """Reparameterization Trick: z = z_mean + exp(0.5 * z_log_var) * epsilon"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# VAE Encoder
encoder_inputs = keras.Input(shape=(784,))
x = layers.Dense(256, activation="relu")(encoder_inputs)
x = layers.Dense(128, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# VAE Decoder
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(128, activation="relu")(latent_inputs)
x = layers.Dense(256, activation="relu")(x)
decoder_outputs = layers.Dense(784, activation="sigmoid")(x)
decoder = Model(latent_inputs, decoder_outputs, name="decoder")

class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            # Binary crossentropy sum over pixels
            bce = keras.losses.binary_crossentropy(data, reconstruction)
            reconstruction_loss = tf.reduce_mean(bce) * 784
            # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(x_train_flat, epochs=5, batch_size=256, verbose=1)

# Sample 5 random points from the standard normal latent space
print("\nSampling 5 random vectors from standard normal latent space N(0, I)...")
random_latent_points = np.random.normal(size=(5, latent_dim))
generated_images = vae.decoder.predict(random_latent_points)

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for k in range(5):
    axes[k].imshow(generated_images[k].reshape(28, 28), cmap='gray')
    axes[k].set_title(f"Sampled Item {k+1}\nz ~ N(0, I)")
    axes[k].axis('off')

plt.suptitle("Task 4: Newly Generated Images from Sampled Latent Vectors via VAE Decoder", fontsize=13)
plt.tight_layout()
task4_path = os.path.join(OUTPUT_DIR, "vae_sampled_digits.png")
plt.savefig(task4_path, dpi=150)
plt.close()
print(f"Task 4 result saved to '{task4_path}'")

print("\n" + "=" * 70)
print("ALL SESSION 2 TASKS SUCCESSFULLY EXECUTED AND SAVED!")
print("=" * 70)

