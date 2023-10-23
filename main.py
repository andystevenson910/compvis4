import cv2
import numpy as np
import matplotlib.pyplot as plt 
import pywt
import skimage.util

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
            if D != 0:
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


###section 4

# 1. Compute Maximum Decomposition Level and Restore Image:
img = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE)
max_level = pywt.dwt_max_level(len(img), 'db2')
coeffs = pywt.wavedec2(img, 'db2', level=max_level)
restored = pywt.waverec2(coeffs, 'db2')

#the choice for the all close function is to resolve an issue that arrises with a rounding difference
#that affects literally 1 pixel ONE PIXEL so i just used all close instead but i stg its one pixel
if np.allclose(img, restored):
    print("The original and the restored images are the same.")
else:
    print("The original and the restored images are different.")

# 2. 3-Level “db2” Wavelet Decomposition:
coeffs_3 = pywt.wavedec2(img, 'db2', level=3)

# a) Set the 16 values of each 4×4 non-overlapping block in the approximation subband as its average.
cA3_avg = coeffs_3[0].copy()
h, w = cA3_avg.shape
for i in range(0, h, 4):
    for j in range(0, w, 4):
        mean_val = np.mean(cA3_avg[i:i+4, j:j+4])
        cA3_avg[i:i+4, j:j+4] = mean_val
coeffs_3_avg = [cA3_avg] + coeffs_3[1:]
restored_avg = pywt.waverec2(coeffs_3_avg, 'db2')

# b) Set the first level horizontal detail coefficients as 0’s.
coeffs_3_h0 = coeffs_3.copy()


#coeffs_3_h0[1] = (coeffs_3_h0[1][0], np.zeros_like(coeffs_3_h0[1][1]), coeffs_3_h0[1][2])
num_rows = len(coeffs_3_h0[1][1])
num_cols = len(coeffs_3_h0[1][1][0]) if num_rows > 0 else 0

# Create a 2D matrix of zeros of the same size
zeros_matrix = [[0] * num_cols for _ in range(num_rows)]

# Update the tuple
coeffs_3_h0[1] = (coeffs_3_h0[1][0], zeros_matrix, coeffs_3_h0[1][2])




restored_h0 = pywt.waverec2(coeffs_3_h0, 'db2')

# c) Set the second level diagonal detail coefficients as 0’s.
# Assuming coeffs_3 is some data structure you've previously defined
coeffs_3_d0 = coeffs_3.copy()

# Get the dimensions of the original 2D matrix
num_rows = len(coeffs_3_d0[2][2])
num_cols = len(coeffs_3_d0[2][2][0]) if num_rows > 0 else 0

# Create a 2D matrix of zeros of the same size
zeros_matrix = [[0] * num_cols for _ in range(num_rows)]

# Update the tuple
coeffs_3_d0[2] = (coeffs_3_d0[2][0], coeffs_3_d0[2][1], zeros_matrix)
restored_d0 = pywt.waverec2(coeffs_3_d0, 'db2')


# d) Set the third level vertical detail coefficients as 0’s.
# Assuming coeffs_3 is some data structure you've previously defined
coeffs_3_v0 = coeffs_3.copy()

# Get the dimensions of the original 2D matrix
num_rows = len(coeffs_3_v0[3][0])
num_cols = len(coeffs_3_v0[3][0][0]) if num_rows > 0 else 0

# Create a 2D matrix of zeros of the same size
zeros_matrix = [[0] * num_cols for _ in range(num_rows)]

# Update the tuple
coeffs_3_v0[3] = (zeros_matrix, coeffs_3_v0[3][1], coeffs_3_v0[3][2])
restored_v0 = pywt.waverec2(coeffs_3_v0, 'db2')

# Display
plt.figure(7), plt.imshow(restored_avg, cmap="gray"), plt.title('Avg Approximation')
plt.figure(8), plt.imshow(restored_h0, cmap="gray"), plt.title('Horizontal Detail as 0s')
plt.figure(9), plt.imshow(restored_d0, cmap="gray"), plt.title('Diagonal Detail as 0s')
plt.figure(10), plt.imshow(restored_v0, cmap="gray"), plt.title('Vertical Detail as 0s')


### section 5
def threshold_coefficients(coeffs, threshold):
    return pywt.threshold(coeffs, threshold, mode='soft')

def compute_threshold(coeffs, M):
    sigma = np.median(np.abs(coeffs)) / 0.6745
    return sigma * np.sqrt(2 * np.log(M))

def denoise_method(image, method):
    coeffs = pywt.wavedec2(image, 'db2', level=3)
    for i in range(1, 4):
        M = len(coeffs[i][0]) * len(coeffs[i][1])
        if method == 1 and i == 1:
            sigma = compute_threshold(coeffs[i][2], M)
        else:
            sigma = compute_threshold(np.concatenate(coeffs[i]), M)
        threshold = sigma * np.sqrt(2 * np.log(M))
        coeffs[i] = tuple(map(lambda x: threshold_coefficients(x, threshold), coeffs[i]))
    return pywt.waverec2(coeffs, 'db2')

image = cv2.imread('Lena.jpg', cv2.IMREAD_GRAYSCALE)
noisy_image = skimage.util.random_noise(image, mode='gaussian', mean = 0, var = 0.01)
cv2.imwrite('NoisyLena.bmp', noisy_image)

denoised_image1 = denoise_method(noisy_image, 1)
denoised_image2 = denoise_method(noisy_image, 2)

plt.figure()
plt.subplot(1, 3, 1), plt.imshow(noisy_image, cmap='gray'), plt.title('Noisy Image')
plt.subplot(1, 3, 2), plt.imshow(denoised_image1, cmap='gray'), plt.title('Denoised Method 1')
plt.subplot(1, 3, 3), plt.imshow(denoised_image2, cmap='gray'), plt.title('Denoised Method 2')
plt.show()
