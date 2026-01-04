import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read image in grayscale
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
n = 1000
D0 = 10

# 2. Compute 2D Fourier Transform
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# 3. Create Butterworth filter mask
rows, cols = img.shape
crow, ccol = rows//2, cols//2  # Center coordinates

# Create distance matrix
u = np.arange(rows) - crow
v = np.arange(cols) - ccol
V, U = np.meshgrid(v, u)
D = np.sqrt(U**2 + V**2)

# Butterworth transfer function
H = 1 / (1 + (D/D0)**(2*n))

# Convert to 2-channel format for OpenCV
H = np.stack([H, H], axis=-1)

# 4. Apply filter
filtered_shift = dft_shift * H

# 5. Inverse Fourier Transform
f_ishift = np.fft.ifftshift(filtered_shift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

# 6. Normalize output
img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Display results
plt.figure(figsize=(8,4))
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title(f'Butterworth Filtered (D0={D0}, n={n})'), plt.xticks([]), plt.yticks([])
plt.show()