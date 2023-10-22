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
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(np.log(1+np.abs(H)), cmap='gray')
plt.title('Gaussian Low-pass Filter')

plt.subplot(1, 3, 3)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')

plt.tight_layout()


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



###section 2 
#prb1

# Load the images
sample_image = cv2.imread('Sample.jpg', cv2.IMREAD_GRAYSCALE)
capital_image = cv2.imread('Capitol.jpg', cv2.IMREAD_GRAYSCALE)

# Apply the 2D Fourier Transform to the images
f_sample = np.fft.fft2(sample_image)
f_capital = np.fft.fft2(capital_image)

# Shift the zero-frequency component to the center of the spectrum
fshift_sample = np.fft.fftshift(f_sample)
fshift_capital = np.fft.fftshift(f_capital)

# Get magnitude and phase for both images
magnitude_spectrum_sample = np.log(np.abs(fshift_sample) + 1)
magnitude_spectrum_capital = np.log(np.abs(fshift_capital) + 1)

phase_spectrum_sample = np.angle(fshift_sample)
phase_spectrum_capital = np.angle(fshift_capital)

# Display the results
plt.figure('Figure 3')

plt.subplot(2, 2, 1), plt.imshow(magnitude_spectrum_sample, cmap='gray')
plt.title('Sample Magnitude Spectrum'), plt.axis('off')

plt.subplot(2, 2, 2), plt.imshow(magnitude_spectrum_capital, cmap='gray')
plt.title('Capital Magnitude Spectrum'), plt.axis('off')

plt.subplot(2, 2, 3), plt.imshow(phase_spectrum_sample, cmap='gray')
plt.title('Sample Phase Spectrum'), plt.axis('off')

plt.subplot(2, 2, 4), plt.imshow(phase_spectrum_capital, cmap='gray')
plt.title('Capital Phase Spectrum'), plt.axis('off')



###prb2

reconstructed_capital_fshift = np.abs(fshift_sample) * np.exp(1j * phase_spectrum_capital)
reconstructed_sample_fshift = np.abs(fshift_capital) * np.exp(1j * phase_spectrum_sample)

# Apply inverse Fourier transform
reconstructed_capital = np.abs(np.fft.ifft2(np.fft.ifftshift(reconstructed_capital_fshift)))
reconstructed_sample = np.abs(np.fft.ifft2(np.fft.ifftshift(reconstructed_sample_fshift)))

# Display the reconstructed images
plt.figure('Figure 4')

plt.subplot(1, 2, 1), plt.imshow(reconstructed_sample, cmap='gray')
plt.title('Reconstructed Sample'), plt.axis('off')

plt.subplot(1, 2, 2), plt.imshow(reconstructed_capital, cmap='gray')
plt.title('Reconstructed Capital'), plt.axis('off')


###section 3



def compute_dft(image):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)
    return dft_shifted

def magnitude_spectrum(dft_shifted):
    magnitude = 20 * np.log(cv2.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1]))
    return magnitude

def locate_largest_magnitudes(magnitude, count):
    # Mask the 3x3 center
    center = np.array(magnitude.shape) // 2
    magnitude[center[0]-1:center[0]+2, center[1]-1:center[1]+2] = 0

    # Get the indices of the largest magnitudes
    indices = np.unravel_index(np.argsort(magnitude.ravel())[-count:], magnitude.shape)
    return indices

def replace_with_neighbors_average(dft_shifted, indices):
    # For each index, replace its value with the average of its 8 neighbors
    for i, j in zip(*indices):
        neighbors = []
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                if x == 0 and y == 0:
                    continue
                neighbors.append(dft_shifted[i + x, j + y])
        dft_shifted[i, j] = np.mean(neighbors, axis=0)
    return dft_shifted

# Load the image
image_path = "boy_noisy.gif" # change this to the path of your image if different
image = plt.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Compute the DFT and magnitude spectrum
dft_shifted = compute_dft(image)
magnitude = magnitude_spectrum(dft_shifted)

# Display the resultant images as per step 6
plt.figure('Figure 6')
max_counts = [2, 3, 5, 6]

for idx, count in enumerate(max_counts, 1):
    indices = locate_largest_magnitudes(magnitude.copy(), count)
    modified_fshift = replace_with_neighbors_average(dft_shifted.copy(), indices)
    
    restored_image = cv2.idft(np.fft.ifftshift(modified_fshift))
    restored_image = cv2.magnitude(restored_image[:, :, 0], restored_image[:, :, 1])
    
    plt.subplot(1, 4, idx)
    plt.imshow(restored_image, cmap='gray')
    plt.title(f'Restored Image ({count} Max Locations)')

plt.tight_layout()
plt.show()
