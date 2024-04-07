'''
    @Name: Felipe Andrade Garcia Tommaselli
    @Number: 11800910
    @Course: SCC0651
    @Year: 2024.1
    @Title: Assign 2- Fourier Transform and Filtering
'''

import numpy as np
import imageio.v3 as iio

def rmse(res_img, ref_img):
    return round(np.sqrt(np.mean((res_img.astype(np.int16) - ref_img.astype(np.int16)) ** 2)), 4)

def fourier_transform(img):
    return np.fft.fftshift(np.fft.fft2(img))

def inverse_fourier_transform(img):
    return np.fft.ifft2(np.fft.ifftshift(img)).real

def low_pass(img, radius):
    rows, cols = img.shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2) <= radius:
                mask[i, j] = 1

    return img * mask

def high_pass(img, radius):
    rows, cols = img.shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2) >= radius:
                mask[i, j] = 1

    return img * mask

def band_stop(img, radius1, radius2):
    rows, cols = img.shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.ones((rows, cols))  

    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            if radius2 <= distance <= radius1:  
                mask[i, j] = 0  

    return img * mask

def laplacian(img): 
    rows, cols = img.shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            mask[i, j] = - 4 * np.pi ** 2 * ((i - center_row) ** 2 + (j - center_col) ** 2)

    return img * mask

def gaussian(img, sigma1, sigma2):
    rows, cols = img.shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            mask[i, j] = np.exp(-((i - center_row) ** 2 / (2 * sigma1 ** 2) + (j - center_col) ** 2 / (2 * sigma2 ** 2)))

    return img * mask

if __name__ == "__main__":
    img = input().rstrip()
    img_ref = input().rstrip()
    filter_id = int(input().rstrip())    

    if filter_id == 0 or filter_id == 1:
        filter_param0 = float(input().rstrip())
    elif filter_id == 2 or filter_id == 4:
        filter_param0 = float(input().rstrip())
        filter_param1 = float(input().rstrip())
    elif filter_id == 3:
        filter_param0, filter_param1 = None, None

    img = iio.imread(img)
    img_ref = iio.imread(img_ref)

    img_spectrum = fourier_transform(img)

    if filter_id == 0:
        res_img_spectrum = low_pass(img_spectrum, filter_param0)
    
    elif filter_id == 1:
        res_img_spectrum = high_pass(img_spectrum, filter_param0)

    elif filter_id == 2:
        res_img_spectrum = band_stop(img_spectrum, filter_param0, filter_param1)
    
    elif filter_id == 3:
        res_img_spectrum = laplacian(img_spectrum)
    
    elif filter_id == 4:
        res_img_spectrum = gaussian(img_spectrum, filter_param0, filter_param1)

    res_img = inverse_fourier_transform(res_img_spectrum)

    res_img = ((res_img - np.min(res_img)) / (np.max(res_img) - np.min(res_img)) * 255).astype(np.uint8)

    print(f"{rmse(res_img, img_ref):.4f}")