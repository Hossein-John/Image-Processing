import cv2
import numpy as np
from scipy import fft

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