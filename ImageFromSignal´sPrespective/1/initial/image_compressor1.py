import cv2
import numpy as np
from scipy import fft
import matplotlib.pyplot as plt

def compress(image_path):
    gray_image_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(float)
    h, w = gray_image_original.shape
    block_size = 32
    gray_image_compressed = gray_image_original.copy()

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = gray_image_original[i:i+block_size, j:j+block_size]
            if block.shape[0] != block_size or block.shape[1] != block_size:
                continue

            fft_block = fft.fft2(block)
            F_max = np.max(np.abs(fft_block))
            tol = 0.001
            threshold = tol * F_max  
            fft_block_compressed = fft_block * (np.abs(fft_block) >= threshold)
            
            gray_image_compressed[i:i+block_size, j:j+block_size] = np.real(fft.ifft2(fft_block_compressed))
    
    gray_image_compressed = np.clip(gray_image_compressed, 0, 255)
    gray_image_compressed = gray_image_compressed.astype(np.uint8)



    return gray_image_compressed


def mse(original, compressed):
    return np.mean((original - compressed) ** 2)


def psnr(original, compressed):
    mse_value = mse(original, compressed)
    if mse_value == 0:
        return float('inf')
    max_I = 255.0
    return 10 * np.log10((max_I ** 2) / mse_value)


gray_image_compressed = compress("./Data/operahall.jpg")
cv2.imwrite("gray_image_compressed.jpg", gray_image_compressed)

gray_image_original = cv2.imread("./Data/operahall.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imwrite("gray_image_original.jpg", gray_image_original)


mse_value = mse(gray_image_original, gray_image_compressed)
psnr_value = psnr(gray_image_original, gray_image_compressed)

print("MSE:", mse_value)
print("PSNR:", psnr_value, "dB")


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(gray_image_original, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(gray_image_compressed, cmap='gray')
plt.title("Compressed Image")
plt.axis('off')

plt.tight_layout()
plt.show()


