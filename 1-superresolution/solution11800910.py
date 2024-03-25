'''
Name: Gianluca Capezzuto Sardinha
NUSP: 11876933
Course Code: SCC0651
Year/Semester: 2024/1
Title: Enhancement and Superresolution
'''

import numpy as np
import imageio.v3 as iio

def rmse(enh_image, ref_image):
    return round(np.sqrt(np.mean((enh_image.astype(np.int16) - ref_image.astype(np.int16)) ** 2)), 4)

def load_images(name):
    images = []

    for type in range(0, 4):
        image = iio.imread(str(name) + str(type) + '.png')
        images.append(image)

    return images

def map_indices(i, j):
    if i % 2 == 0 and j % 2 == 0:
        return 0
    elif i % 2 == 1 and j % 2 == 0:
        return 1
    elif i % 2 == 0 and j % 2 == 1:
        return 2
    elif i % 2 == 1 and j % 2 == 1:
        return 3

def superresolution(low_images, high_image):
    h, w = high_image.shape
    low_image_upscaled = np.zeros((h, w)) 

    for i in range(0, h):
        for j in range(0, w):
            low_image_upscaled[i, j] = low_images[map_indices(i, j)][i // 2, j // 2]

    low_image_upscaled = low_image_upscaled.astype(np.uint8)

    return rmse(low_image_upscaled, high_image)

def histogram(image, no_levels = 256):
    hist = np.zeros(no_levels).astype(int)

    for i in range(no_levels):
        no_pixel_value_i = np.sum(image == i)
    
        hist[i] = no_pixel_value_i

    return hist

def single_cumulative_histogram(images, no_levels = 256):
    hist_transf_list = []
    image_eq_list = []

    for image in images:
        hist = histogram(image, no_levels)

        histC = np.zeros(no_levels).astype(int)
        histC[0] = hist[0]

        for i in range(1, no_levels):
            histC[i] = hist[i] + histC[i - 1]

        hist_transf = np.zeros(no_levels).astype(np.uint8)

        N, M = image.shape
        image_eq = np.zeros([N, M]).astype(np.uint8)

        for z in range(no_levels):
            s = ((no_levels - 1) / float(M * N)) * histC[z]
            hist_transf[z] = s

            image_eq[np.where(image == z)] = s

        hist_transf_list.append(hist_transf)
        image_eq_list.append(image_eq)

    return image_eq_list, hist_transf_list

def joint_cumulative_histogram(images, no_levels = 256):
    hist_images = []    
    hist_transf_list = []
    image_eq_list = []

    for image in images:
        hist = histogram(image, no_levels)

        histC = np.zeros(no_levels).astype(int)
        histC[0] = hist[0]

        for i in range(1, no_levels):
            histC[i] = hist[i] + histC[i - 1]

        hist_images.append(histC)

    histC = np.zeros(no_levels).astype(int)

    for i in range(0, 4):
        histC += hist_images[i]

    for image in images:
        hist_transf = np.zeros(no_levels).astype(np.uint8)

        N, M = image.shape
        image_eq = np.zeros([N, M]).astype(np.uint8)

        for z in range(no_levels):
            s = ((1/4) * (no_levels - 1) / float(M * N)) * histC[z]
            hist_transf[z] = s

            image_eq[np.where(image == z)] = s

        hist_transf_list.append(hist_transf)
        image_eq_list.append(image_eq)

    return image_eq_list, hist_transf_list

def gamma_correction(images, gamma):
    image_eq_list = []

    for image in images:
        image_eq = (255 * ((image / 255) ** (1 / gamma))).astype(np.uint8) 
        image_eq_list.append(image_eq)

    return image_eq_list

if __name__ == "__main__":
    imagelow = input().rstrip()
    imagehigh = input().rstrip()
    f = int(input().rstrip())
    gamma = float(input().rstrip())

    low_images = load_images(imagelow)
    imagehigh = iio.imread(imagehigh)

    if f == 0:
        error = superresolution(low_images, imagehigh)

        print(error)
    elif f == 1:
        enhanced_images, _ = single_cumulative_histogram(low_images)

        error = superresolution(enhanced_images, imagehigh)

        print(error)
    elif f == 2:
        enhanced_images, _ = joint_cumulative_histogram(low_images)

        error = superresolution(enhanced_images, imagehigh)

        print(error)
    elif f == 3:
        enhanced_images = gamma_correction(low_images, gamma)

        error = superresolution(enhanced_images, imagehigh)

        print(error)