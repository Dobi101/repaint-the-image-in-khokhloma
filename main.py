import copy
import numpy as np
import cv2 as cv


def load_image(directory):
    images = cv.imread(directory)
    scale_percent = 30  # percent of original size
    width = int(images.shape[1] * scale_percent / 100)
    height = int(images.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv.resize(images, dim, interpolation=cv.INTER_AREA)


def k_means(img, k):
    z = img.reshape((-1, 3))
    z = np.float32(z)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv.kmeans(z, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape(img.shape)


def fill_color(image):
    x, y, z = image.shape
    for i in range(x):
        for j in range(y):
            if len(c1) == 0:
                c1.append([i, j])
            elif np.array_equal(image[(c1[0][0], c1[0][1])], image[i, j]):
                c1.append([i, j])
            elif len(c2) == 0:
                c2.append([i, j])
            elif np.array_equal(image[(c2[0][0], c2[0][1])], image[i, j]):
                c2.append([i, j])
            elif len(c3) == 0:
                c3.append([i, j])
            elif np.array_equal(image[(c3[0][0], c3[0][1])], image[i, j]):
                c3.append([i, j])
            elif len(c4) == 0:
                c4.append([i, j])
            elif np.array_equal(image[(c4[0][0], c4[0][1])], image[i, j]):
                c4.append([i, j])
            elif len(c5) == 0:
                c5.append([i, j])
            elif np.array_equal(image[(c5[0][0], c5[0][1])], image[i, j]):
                c5.append([i, j])
            elif len(c6) == 0:
                c6.append([i, j])
            elif np.array_equal(image[(c6[0][0], c6[0][1])], image[i, j]):
                c6.append([i, j])


def v1(image):
    for i in c1:
        image[i[0], i[1]] = colors[0]
    for i in c2:
        image[i[0], i[1]] = colors[1]
    for i in c3:
        image[i[0], i[1]] = colors[2]
    for i in c4:
        image[i[0], i[1]] = colors[3]
    for i in c5:
        image[i[0], i[1]] = colors[4]
    for i in c6:
        image[i[0], i[1]] = colors[5]
    return image


def v2(image):
    for i in c1:
        image[i[0], i[1]] = colors[4]
    for i in c2:
        image[i[0], i[1]] = colors[0]
    for i in c3:
        image[i[0], i[1]] = colors[5]
    for i in c4:
        image[i[0], i[1]] = colors[3]
    for i in c5:
        image[i[0], i[1]] = colors[1]
    for i in c6:
        image[i[0], i[1]] = colors[2]
    return image


def v3(image):
    for i in c1:
        image[i[0], i[1]] = colors[5]
    for i in c2:
        image[i[0], i[1]] = colors[1]
    for i in c3:
        image[i[0], i[1]] = colors[3]
    for i in c4:
        image[i[0], i[1]] = colors[2]
    for i in c5:
        image[i[0], i[1]] = colors[4]
    for i in c6:
        image[i[0], i[1]] = colors[0]
    return image


def v4(image):
    for i in c1:
        image[i[0], i[1]] = colors[0]
    for i in c2:
        image[i[0], i[1]] = colors[5]
    for i in c3:
        image[i[0], i[1]] = colors[3]
    for i in c4:
        image[i[0], i[1]] = colors[1]
    for i in c5:
        image[i[0], i[1]] = colors[4]
    for i in c6:
        image[i[0], i[1]] = colors[2]
    return image


def v5(image):
    for i in c1:
        image[i[0], i[1]] = colors[5]
    for i in c2:
        image[i[0], i[1]] = colors[0]
    for i in c3:
        image[i[0], i[1]] = colors[1]
    for i in c4:
        image[i[0], i[1]] = colors[4]
    for i in c5:
        image[i[0], i[1]] = colors[2]
    for i in c6:
        image[i[0], i[1]] = colors[3]
    return image


def show():
    cv.imshow('original', img)
    cv.imshow('k-means', k_res)
    cv.imshow('1', cv1)
    cv.imshow('2', cv2)
    cv.imshow('3', cv3)
    cv.imshow('4', cv4)
    cv.imshow('5', cv5)
    cv.waitKey(0)
    cv.destroyAllWindows()


c1 = []
c2 = []
c3 = []
c4 = []
c5 = []
c6 = []
colors = [[11, 11, 11], [36, 202, 232], [34, 40, 228], [27,31,161], [26,57,96], [25,126,230]]
img = load_image('im.jpg')
k_res = k_means(img, 6)
buffer = copy.copy(k_res)
fill_color(buffer)
b1 = copy.copy(buffer)
b2 = copy.copy(buffer)
b3 = copy.copy(buffer)
b4 = copy.copy(buffer)
b5 = copy.copy(buffer)
cv1 = v1(b1)
cv2 = v2(b2)
cv3 = v3(b3)
cv4 = v4(b4)
cv5 = v5(b5)
show()
