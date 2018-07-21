from skimage import io,img_as_ubyte,img_as_float
import numpy as np
from copy import deepcopy
from PIL import Image,ImageDraw,ImageFont
import matplotlib.pyplot as plt

def max_scaling(img):
    dim = img.shape
    for i in xrange(0,dim[0]):
        for j in xrange(0,dim[1]):
            if img[i,j] > 255:
                img[i,j] = 255
    return img

def add_watermark(img,mark):
    nimg = deepcopy(img)
    dim = mark.shape
    for i in xrange(0,dim[0]):
        for j in xrange(0,dim[1]):
            if mark[i,j] <= 235:
                nimg[i,j] = 0
    return nimg

def rgb2gray(img):
    ndim = img.shape
    nimg = np.zeros([ndim[0],ndim[1]],dtype='float')
    for row in xrange(0,ndim[0]):
        for col in xrange(0,ndim[1]):
            nimg[row,col] = int(sum([(0.2126 * img[row,col,0])+                                      (0.7152 * img[row,col,1])+(0.0722 * img[row,col,2])]))
    return nimg

def extract_invisible_watermark(img):
    img = np.array(img,dtype='int')
    dim = img.shape
    wtimg = np.zeros([dim[0],dim[1]])
    for i in xrange(0,dim[0]):
        for j in xrange(0,dim[1]):
            if img[i,j] % 2 != 1:
                wtimg[i,j] = 0
            else:
                wtimg[i,j] = 255
    return np.array(wtimg,dtype='float')

def add_invisible_watermark(img,wimg):
    binwt = binary(wimg)
    nimg = deepcopy(img)
    nimg = np.array(nimg,dtype='int')
    dim = img.shape
    for i in xrange(0,dim[0]):
        for j in xrange(0,dim[1]):
            if int(binwt[i,j]) == 255:
                if nimg[i,j] % 2 != 1:
                    nimg[i,j] = nimg[i,j] + 1
            else:
                if nimg[i,j] % 2 == 1:
                    nimg[i,j] = nimg[i,j] - 1
    return np.array(nimg,dtype='float')

def binary(img):
    bimg = deepcopy(img)
    dim = img.shape
    for i in xrange(0,dim[0]):
        for j in xrange(0,dim[1]):
            if img[i,j] <= 235:
                bimg[i,j] = 0
            else:
                bimg[i,j] = 255
    return bimg

img = io.imread("ronaldo.jpg")
img = rgb2gray(img)
info_img = io.imread("clipart.png")
info_img = rgb2gray(info_img)

show_img = plt.imshow(img/img.max(),cmap='gray')
plt.title("Original Image")
plt.show()

pimg = img[64:192,64:192]
show_pimg = plt.imshow(pimg/pimg.max(),cmap='gray')
plt.title("128 x 128 pixel")
plt.show()

pimg = max_scaling(pimg + 60)
show_pimg = plt.imshow(pimg/pimg.max(),cmap='gray')
plt.title("128 x 128 pixel constant added 60")
plt.show()

mask_img = deepcopy(img)
mask_img[64:192,64:192] = max_scaling(mask_img[64:192,64:192] + 60)
show_pimg = plt.imshow(mask_img/mask_img.max(),cmap='gray')
plt.title("Original Pixel Masked")
plt.show()


plt.figure(figsize=(20,20))
splots = [521,522,523,524,525]
for (index,i) in enumerate([0.1,0.5,1.0,1.5,2.0]):
    mask_img = deepcopy(img)
    mask_img[64:192,64:192] = max_scaling(mask_img[64:192,64:192] * i)
    plt.subplot(splots[index])
    show_pimg = plt.imshow(mask_img/mask_img.max(),cmap='gray')
    plt.title("constant = " + str(i))
plt.show()


show_wtimg = plt.imshow(info_img/info_img.max(),cmap='gray')
plt.title("Image for Watermarking")
plt.show()


wimg = add_watermark(img,info_img)
show_wimg = plt.imshow(wimg/wimg.max(),cmap='gray')
plt.title("Visible Watermarked Image")
plt.show()


plt.figure(figsize = (20,10))
plt.subplot(221)
iv_img = add_invisible_watermark(img,info_img)
show_ivimg = plt.imshow(iv_img/iv_img.max(),cmap='gray')
plt.title("Invisible Watermarked Image")
plt.subplot(222)
ex_wt = extract_invisible_watermark(iv_img)
show_exwt = plt.imshow(ex_wt/ex_wt.max(),cmap='gray')
plt.title("Extracted Watermarked Image")
plt.show()
