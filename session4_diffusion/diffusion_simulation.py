"""
SESSION 4: Diffusion Models Simulation
-------------------------------------
1. Forward Diffusion Process: Takes a grayscale image and adds Gaussian noise over 10 discrete steps,
   saving each noisy image to disk.
2. Reverse Diffusion Approximation: Applies filters (median and Gaussian blur) to approximate denoising
   and compares the reconstructed output with the original image.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_sample_grayscale_image(filepath="session4_diffusion/sample_input.png"):
    """Creates a distinct 64x64 grayscale test image with geometric patterns and text."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    img = np.zeros((64, 64), dtype=np.uint8)
    
    # Draw background shapes (circle, square, diagonal line)
    cv2.circle(img, (32, 32), 20, 220, -1)
    cv2.rectangle(img, (12, 12), (32, 32), 100, -1)
    cv2.line(img, (8, 56), (56, 8), 255, 3)
    cv2.putText(img, "AI", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 40, 2)
    
    cv2.imwrite(filepath, img)
    print(f"Created base test image at '{filepath}'")
    return img


def simulate_forward_diffusion(image_path, output_dir="session4_diffusion/forward_steps", num_steps=10):
    """
    Simulates the forward diffusion process (Markovian noise injection).
    q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1}, beta_t * I)
    Adds progressive Gaussian noise across 10 discrete time steps and saves each frame to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    # Normalize to [0, 1] float
    x = img.astype(np.float32) / 255.0
    
    # Linear noise schedule (beta from 0.05 to 0.5)
    betas = np.linspace(0.05, 0.5, num_steps)
    
    noisy_images = [x.copy()]
    # Save initial step 0 (clean image)
    step_0_path = os.path.join(output_dir, "step_00.png")
    cv2.imwrite(step_0_path, (x * 255).astype(np.uint8))
    
    current_x = x.copy()
    print(f"Starting Forward Diffusion over {num_steps} steps...")
    for t in range(1, num_steps + 1):
        beta_t = betas[t - 1]
        noise = np.random.normal(loc=0.0, scale=1.0, size=current_x.shape)
        
        # Closed-form or step-by-step update: sqrt(1 - beta) * x_{t-1} + sqrt(beta) * noise
        current_x = np.sqrt(1.0 - beta_t) * current_x + np.sqrt(beta_t) * noise
        current_x = np.clip(current_x, 0.0, 1.0)
        
        noisy_images.append(current_x.copy())
        step_filename = os.path.join(output_dir, f"step_{t:02d}.png")
        cv2.imwrite(step_filename, (current_x * 255).astype(np.uint8))
        print(f"  Step {t:02d}/{num_steps}: Variance injected beta={beta_t:.3f} -> Saved {step_filename}")

    # Plot sequence summary
    plt.figure(figsize=(15, 3))
    for idx, frame in enumerate(noisy_images):
        plt.subplot(1, num_steps + 1, idx + 1)
        plt.imshow(frame, cmap='gray')
        plt.title(f"t={idx}")
        plt.axis('off')
    plt.suptitle("Forward Diffusion: Gradual Latent Gaussian Noise Injection (t=0 to t=10)")
    plt.tight_layout()
    summary_plot = os.path.join(output_dir, "forward_diffusion_progression.png")
    plt.savefig(summary_plot, dpi=150)
    plt.close()
    print(f"Saved forward progression overview to '{summary_plot}'")

    return noisy_images


def approximate_reverse_diffusion(original_image_path, noisy_image_path, output_path="session4_diffusion/outputs/reverse_diffusion_comparison.png"):
    """
    Simulates reverse diffusion: takes a noisy image and applies denoising filters
    (cv2.medianBlur and cv2.GaussianBlur) to approximate noise estimation & removal,
    then visually compares the denoised images with the ground-truth original.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    orig = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    noisy = cv2.imread(noisy_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Denoising approaches
    # 1. Median blur: effective at non-linear salt/pepper and outlier noise suppression
    median_denoised = cv2.medianBlur(noisy, 5)
    
    # 2. Gaussian blur: linear low-pass smoothing
    gaussian_denoised = cv2.GaussianBlur(noisy, (5, 5), sigmaX=1.5)
    
    # 3. Non-local means denoising (advanced OpenCV denoising)
    nlm_denoised = cv2.fastNlMeansDenoising(noisy, None, h=25, templateWindowSize=7, searchWindowSize=21)
    
    # Calculate MSE and PSNR metrics against original
    def compute_psnr(img1, img2):
        mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
        if mse == 0:
            return float('inf'), 0
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        return psnr, mse

    psnr_noisy, mse_noisy = compute_psnr(orig, noisy)
    psnr_median, mse_med = compute_psnr(orig, median_denoised)
    psnr_gauss, mse_gauss = compute_psnr(orig, gaussian_denoised)
    psnr_nlm, mse_nlm = compute_psnr(orig, nlm_denoised)

    # Plot comparison side by side
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    
    axes[0].imshow(orig, cmap='gray')
    axes[0].set_title("Original Image (t=0)\n(Ground Truth)")
    axes[0].axis('off')
    
    axes[1].imshow(noisy, cmap='gray')
    axes[1].set_title(f"Noisy Image (t=5)\nPSNR: {psnr_noisy:.2f} dB")
    axes[1].axis('off')
    
    axes[2].imshow(median_denoised, cmap='gray')
    axes[2].set_title(f"Median Filter Denoised\nPSNR: {psnr_median:.2f} dB")
    axes[2].axis('off')
    
    axes[3].imshow(gaussian_denoised, cmap='gray')
    axes[3].set_title(f"Gaussian Filter Denoised\nPSNR: {psnr_gauss:.2f} dB")
    axes[3].axis('off')
    
    axes[4].imshow(nlm_denoised, cmap='gray')
    axes[4].set_title(f"NLM Denoised Filter\nPSNR: {psnr_nlm:.2f} dB")
    axes[4].axis('off')
    
    plt.suptitle("Reverse Diffusion Approximation: Classical Denoising vs True Signal", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved reverse diffusion comparison to '{output_path}'")
    print(f"Metrics Summary:")
    print(f"  - Noisy (Step 5):   MSE={mse_noisy:.2f}, PSNR={psnr_noisy:.2f} dB")
    print(f"  - Median Blur:      MSE={mse_med:.2f}, PSNR={psnr_median:.2f} dB")
    print(f"  - Gaussian Blur:    MSE={mse_gauss:.2f}, PSNR={psnr_gauss:.2f} dB")
    print(f"  - Non-Local Means:  MSE={mse_nlm:.2f}, PSNR={psnr_nlm:.2f} dB")


if __name__ == "__main__":
    base_img_path = "session4_diffusion/sample_input.png"
    create_sample_grayscale_image(base_img_path)
    simulate_forward_diffusion(base_img_path)
    # Use step 05 for reverse diffusion demonstration
    noisy_img_path = "session4_diffusion/forward_steps/step_05.png"
    approximate_reverse_diffusion(base_img_path, noisy_img_path)

