'''
    @Name: Felipe Andrade Garcia Tommaselli
    @Number: 11800910
    @Course: SCC0651
    @Year: 2024.1
    @Title: Assign 2- Fourier Transform and Filtering
'''

import numpy as np
import imageio.v3 as iio


def frequency_transform(image):
    return np.fft.fftshift(np.fft.fft2(image))

def inverse_frequency_transform(image):
    return np.fft.ifft2(np.fft.ifftshift(image)).real

def apply_low_pass_filter(image, cutoff_radius):
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2) <= cutoff_radius:
                mask[i, j] = 1

    return image * mask

def apply_high_pass_filter(image, cutoff_radius):
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2) >= cutoff_radius:
                mask[i, j] = 1

    return image * mask

def apply_band_stop_filter(image, low_cutoff_radius, high_cutoff_radius):
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.ones((rows, cols))  

    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            if high_cutoff_radius <= distance <= low_cutoff_radius:  
                mask[i, j] = 0  

    return image * mask

def apply_laplacian_filter(image): 
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            mask[i, j] = - 4 * np.pi ** 2 * ((i - center_row) ** 2 + (j - center_col) ** 2)

    return image * mask

def apply_gaussian_filter(image, sigma_x, sigma_y):
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            mask[i, j] = np.exp(-((i - center_row) ** 2 / (2 * sigma_x ** 2) + (j - center_col) ** 2 / (2 * sigma_y ** 2)))

    return image * mask


if __name__ == "__main__":
    image_input = input().rstrip()
    image_reference = input().rstrip()
    chosen_filter = int(input().rstrip())    

    if chosen_filter in [0, 1]:
        param_alpha = float(input().rstrip())
    elif chosen_filter in [2, 4]:
        param_alpha = float(input().rstrip())
        param_beta = float(input().rstrip())
    elif chosen_filter == 3:
        param_alpha, param_beta = None, None

    image_input = iio.imread(image_input)
    image_reference = iio.imread(image_reference)

    image_spectrum = frequency_transform(image_input)

    if chosen_filter == 0:
        result_spectrum = apply_low_pass_filter(image_spectrum, param_alpha)
    elif chosen_filter == 1:
        result_spectrum = apply_high_pass_filter(image_spectrum, param_alpha)
    elif chosen_filter == 2:
        result_spectrum = apply_band_stop_filter(image_spectrum, param_alpha, param_beta)
    elif chosen_filter == 3:
        result_spectrum = apply_laplacian_filter(image_spectrum)
    elif chosen_filter == 4:
        result_spectrum = apply_gaussian_filter(image_spectrum, param_alpha, param_beta)

    result_image = inverse_frequency_transform(result_spectrum)
    result_image = ((result_image - np.min(result_image)) / (np.max(result_image) - np.min(result_image)) * 255).astype(np.uint8)

    final = round(np.sqrt(np.mean((result_image.astype(np.int16) - image_reference.astype(np.int16)) ** 2)), 4)
    print(f"{final:.4f}")
