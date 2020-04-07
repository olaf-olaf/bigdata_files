import skimage
import numpy as np
import matplotlib.pyplot as plt
import pywt
from PIL import Image, ImageCms
import cv2


# Perceptual image hash based on the work of
#  Vadlamudi, Lokanadham Naidu and Vaddella, Rama and Devara, Vasumathi


def sift_dwt_hash(image_path):
    # In this step, the input RGB color image () is resized to  pixels using bi-cubic interpolation
    img = Image.open(image_path).convert("RGB")
    img = img.resize((512,512),Image.BICUBIC)
    img = np.array(img)

    # Convert to Lab
    img = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)

    # # Gaussian low pas filter
    img = np.array(img)
    kernel = np.ones((5,5),np.float32)/25
    img = cv2.filter2D(img,-1,kernel)


    L,A,B=cv2.split(img)

    # # # Get keypoints from the L dimension and select the n best keypoints measured in SIFT algorithm as the local contrast
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=16)
    key_points, des1 = sift.detectAndCompute(L,None)

    # Get kernel of P*P around keypoints
    rois = []
    P = 64
    for kp in key_points:
        y = int(key_points[0].pt[1])
        x = int(key_points[0].pt[0])
        up = int(y - (P/2))
        low = int(y + (P/2))
        left = int(x - (P/2))
        right = int(x + (P/2))
        cropped_img = L[up:low, left:right]


        rois.append(cropped_img)

    # Get DWT representation of the rois
    dwt_representations = []
    level = 2
    for roi in rois:
        dwt_res = pywt.wavedec2(roi,'db1', level = level)
        cA = dwt_res[0]
        dwt_representations.append(cA)
    len(dwt_representations)
    # print(dwt_representations[0].shape)

    # reshape the DWT representation
    Q = len(dwt_representations) * (P / 2**level)
    R = (P/2**level)
    dwt_representations[0].shape
    qr_matrix = dwt_representations[0]
    i = 0
    for dwt in dwt_representations[1:]:
            qr_matrix = np.concatenate((qr_matrix,dwt), axis = 0)


    # # # Get the row wise averages
    row_wise_averages = []
    for row in qr_matrix:
        row_wise_averages.append(np.mean(row))
    row_wise_averages

    # EXPERIMENT: subtract minimum value
    minimum = min(row_wise_averages)
    placeholder = []
    for element in row_wise_averages:
        element = element - minimum
        placeholder.append(element)
    row_wise_averages = placeholder

    # TODO: GET THE RIGHT RANDOM PERMUTATION
    # row_wise_averages = np.random.permutation(row_wise_averages)

    # MAX average the list
    row_wise_averages  = [float(i)/max(row_wise_averages) for i in row_wise_averages]

    # Convert to bit hash
    hash_result = []
    for i in row_wise_averages:
        if i > 0.5:
            hash_result.append(1)
        else:
            hash_result.append(0)
    hash_string = ''
    for hash_element in hash_result:
        hash_string += str(hash_element)
    # return string
    return hash_string
