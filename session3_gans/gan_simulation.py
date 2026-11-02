"""
SESSION 3: Generative Adversarial Networks (GANs) Simulation
-----------------------------------------------------------
1. simple_generator(): Takes a random vector and returns a 28x28 grayscale image (0 to 1).
2. Discriminator: Class with discriminate(image) method returning probability score (0 to 1).
3. Adversarial Training Loop: Simulates one cycle of generation, discrimination, and evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def simple_generator(latent_vector=None, latent_dim=100):
    """
    Takes a random noise vector and transforms it into a 28x28 grayscale array (values 0.0 to 1.0).
    Simulates a generative mapping (e.g., producing a digit-like stroke pattern).
    """
    if latent_vector is None:
        latent_vector = np.random.normal(loc=0.0, scale=1.0, size=(latent_dim,))
    
    # Deterministic pseudo-neural projection using a fixed random seed projection matrix
    # to simulate weights of a generator network
    np.random.seed(int(abs(latent_vector[0]) * 1000) % (2**31 - 1))
    
    # Create a 28x28 base canvas
    base = np.zeros((28, 28), dtype=np.float32)
    
    # Synthesize digit-like strokes based on latent vector components
    # Center stroke simulating a digit like '1', '0', or '7'
    center_x = int(14 + np.clip(latent_vector[1] * 3, -6, 6))
    center_y = int(14 + np.clip(latent_vector[2] * 3, -6, 6))
    radius = int(5 + np.clip(abs(latent_vector[3]) * 3, 2, 8))
    
    y, x = np.ogrid[:28, :28]
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Ring / stroke structure
    stroke = np.exp(-((dist_from_center - radius)**2) / 6.0)
    
    # Add some smooth low-frequency noise from latent vector
    noise_pattern = np.random.uniform(0.0, 0.15, size=(28, 28))
    
    synthetic_image = stroke * 0.8 + noise_pattern
    synthetic_image = np.clip(synthetic_image, 0.0, 1.0)
    
    return synthetic_image


class Discriminator:
    """
    Discriminator model that evaluates a 28x28 image and returns a probability [0.0, 1.0]
    representing whether the image is 'real' (close to 1.0) or 'fake' (close to 0.0).
    Uses feature heuristics: mean pixel intensity, contrast, and center concentration.
    """
    def __init__(self, threshold=0.18):
        self.threshold = threshold

    def discriminate(self, image: np.ndarray) -> float:
        """
        Calculates probability score of image being 'real'.
        Real MNIST digits typically have dark background with focused bright strokes in the center.
        """
        assert image.shape == (28, 28), "Image must be of shape 28x28"
        
        # Metric 1: Mean pixel value (Real digits usually have mean between 0.10 and 0.28)
        mean_val = np.mean(image)
        
        # Metric 2: Central focus (ratio of center 14x14 energy vs border energy)
        center_region = image[7:21, 7:21]
        center_mean = np.mean(center_region)
        border_mean = (np.sum(image) - np.sum(center_region)) / (28*28 - 14*14)
        
        # If center is significantly brighter than border and mean is within natural range
        contrast_ratio = center_mean / (border_mean + 1e-5)
        
        # Compute heuristic probability via sigmoid
        # High center contrast + moderate mean indicates real handwritten digit
        score = 1.0 / (1.0 + np.exp(-(contrast_ratio - 2.5) * 1.5))
        
        # Penalize if whole image is uniform noise or too washed out
        if mean_val < 0.05 or mean_val > 0.45:
            score *= 0.5
            
        return float(np.clip(score, 0.01, 0.99))


def simulate_adversarial_cycle():
    """
    Simulates one cycle of the adversarial training loop:
    1. Generator creates a fake image from latent noise.
    2. Discriminator inspects the fake image and outputs probability score.
    3. Evaluates if Discriminator correctly flagged it as fake.
    """
    print("=" * 60)
    print("SIMULATING ADVERSARIAL CYCLE (GENERATOR vs DISCRIMINATOR)")
    print("=" * 60)
    
    # 1. Generator step
    latent_z = np.random.normal(loc=0.0, scale=1.0, size=(100,))
    fake_img = simple_generator(latent_z)
    print(f"[Generator] Generated fake image with shape {fake_img.shape}, min={fake_img.min():.3f}, max={fake_img.max():.3f}, mean={fake_img.mean():.3f}")
    
    # 2. Discriminator step
    discriminator = Discriminator()
    fake_score = discriminator.discriminate(fake_img)
    print(f"[Discriminator] Probability image is REAL: {fake_score:.4f} (Score for Fake: {1.0 - fake_score:.4f})")
    
    # 3. Decision threshold (0.5)
    is_classified_as_real = fake_score >= 0.5
    decision = "REAL" if is_classified_as_real else "FAKE"
    correct_detection = (decision == "FAKE")
    
    print("-" * 60)
    print(f"Prediction: Image classified as '{decision}'")
    if correct_detection:
        print("[SUCCESS] The Discriminator correctly identified the fake image!")
        print("Adversarial Feedback: Generator loss is HIGH. Generator must update weights to fool Discriminator.")
    else:
        print("[TRICKED] The Generator successfully fooled the Discriminator into thinking it is REAL!")
        print("Adversarial Feedback: Discriminator loss is HIGH. Discriminator must update weights to detect this flaw.")
    print("=" * 60)

    # Save output visualization
    os.makedirs("session3_gans/outputs", exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(fake_img, cmap='gray')
    plt.title(f"Generated Fake Image\nDiscriminator Real Score: {fake_score:.3f} -> '{decision}'")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("session3_gans/outputs/simulated_fake_image.png", dpi=150)
    plt.close()
    print("Saved generated image visualization to 'session3_gans/outputs/simulated_fake_image.png'")


if __name__ == "__main__":
    simulate_adversarial_cycle()

