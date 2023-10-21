import cv2
import numpy as np
import matplotlib.pyplot as plt


##Problem 1
def gaussian_low_pass(u_size, v_size, sigma1, sigma2):
    center_u = u_size // 2
    center_v = v_size // 2
    H = np.zeros((u_size, v_size))
    
    for u in range(u_size):
        for v in range(v_size):
            D_sq = ((u - center_u)**2 / sigma1**2) + ((v - center_v)**2 / sigma2**2)
            H[u, v] = np.exp(-D_sq / 2)
            
    return H

# Load the image in grayscale
image = cv2.imread('Sample.jpg', cv2.IMREAD_GRAYSCALE)

# Compute the 2D FFT of the image
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)

# Design the Gaussian filter
u, v = image.shape
H = gaussian_low_pass(u, v, 20, 70)

# Apply the filter
filtered_fshift = fshift * H

# Compute the inverse FFT to get the filtered image
filtered_f = np.fft.ifftshift(filtered_fshift)
filtered_image = np.abs(np.fft.ifft2(filtered_f))

# Display
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='grey')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(np.log(1+np.abs(H)), cmap='grey')
plt.title('Gaussian Low-pass Filter')

plt.subplot(1, 3, 3)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')

plt.tight_layout()
plt.show()


#problem 2



def butterworth_high_pass(u_size, v_size, D0, n):
    center_u = u_size // 2
    center_v = v_size // 2
    H = np.zeros((u_size, v_size))
    
    for u in range(u_size):
        for v in range(v_size):
            D = np.sqrt((u - center_u)**2 + (v - center_v)**2)
            H[u, v] = 1 / (1 + (D0 / D)**(2*n))
            
    return H

# Load the image in grayscale
image = cv2.imread('Sample.jpg', cv2.IMREAD_GRAYSCALE)

# Compute the 2D FFT of the image
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)

# Design the Butterworth filter
u, v = image.shape
H = butterworth_high_pass(u, v, 50, 2)

# Apply the filter
filtered_fshift = fshift * H

# Compute the inverse FFT to get the filtered image
filtered_f = np.fft.ifftshift(filtered_fshift)
filtered_image = np.abs(np.fft.ifft2(filtered_f))

# Display
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(H, cmap='gray')
plt.title('Butterworth High-pass Filter')

plt.subplot(1, 3, 3)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')

plt.tight_layout()
plt.show()


