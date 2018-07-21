from skimage import io,data
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

def rgb2gray(img):
    ndim = img.shape
    nimg = np.zeros([ndim[0],ndim[1]],dtype='float')
    for row in xrange(0,ndim[0]):
        for col in xrange(0,ndim[1]):
            nimg[row,col] = int(sum([(0.2126 * img[row,col,0])+(0.7152 * img[row,col,1])+(0.0722 * img[row,col,2])]))
    return nimg

def convolve(nimg,W=[],type = 'l'):
    ndim = nimg.shape
    if len(W) == 0:
        W = np.zeros([3,3])
    W = np.rot90(W,2)
    newrow = np.zeros(ndim[1])
    newcol = np.zeros(ndim[0]+2*int(len(W)/2)).reshape(ndim[0]+2*int(len(W)/2),1)
    nimg = np.vstack([newrow for i in xrange(0,int(len(W)/2))]+[nimg]+[newrow for i in xrange(0,int(len(W)/2))])
    nimg = np.hstack([newcol[:None] for i in xrange(0,int(len(W)/2))]+[nimg]+[newcol[:None] for i in xrange(0,int(len(W)/2))])
    tnimg = deepcopy(nimg)
    for i in xrange(int(len(W)/2),ndim[1]+1):
        for j in xrange(int(len(W)/2),ndim[0]+1):
            elmnts = []
            if type == 'nl':
                for m in xrange(-int(len(W)/2),int(len(W)/2)+1):
                    for n in xrange(-int(len(W)/2),int(len(W)/2)+1):
                        elmnts.append(nimg[i+m,j+n])
                tnimg[i,j] = np.median(elmnts)
            else:
                for (midx,m) in enumerate(xrange(-int(len(W)/2),int(len(W)/2)+1)):
                    for (nidx,n) in enumerate(xrange(-int(len(W)/2),int(len(W)/2)+1)):
                        elmnts.append(nimg[i+m,j+n]*W[midx,nidx])
                tnimg[i,j] = sum(elmnts)
    tnimg = np.delete(tnimg,tuple([i for i in xrange(0,int(len(W)/2))]+[tnimg.shape[0]-(i+1) for i in xrange(0,int(len(W)/2))]),axis=0)
    tnimg = np.delete(tnimg,tuple([i for i in xrange(0,int(len(W)/2))]+[tnimg.shape[1]-(i+1) for i in xrange(0,int(len(W)/2))]),axis=1)
    return tnimg

def scaling(nimg):
    K = 255
    nimg = nimg - nimg.min()
    nimg = K * (nimg  / nimg.max())
    return nimg


img = io.imread("ronaldo.jpg")
img = rgb2gray(img)
info_img = io.imread("clipart.png")
info_img = rgb2gray(info_img)

show_img = plt.imshow(img/img.max(),cmap='gray')
plt.title("Original Image")
plt.show()


C = 1.0
ishape = img.shape
avg_filter = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])/25.0
lplc_filter = np.array([[0,1,0],[1,-4,1],[0,1,0]])
gx = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
gy = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])


low_img = convolve(img,W = avg_filter)
show_lowpass = plt.imshow(low_img/low_img.max(),cmap = 'gray')
plt.title("5 X 5 Average Mask Blurred Image")
plt.show()


plt.figure(figsize=(20,10))
sobel_img = scaling(convolve(img,W = gx) + convolve(img,W = gy))
plt.subplot(221)
plt.title("Sobel Filter Applied")
show_sbimg = plt.imshow(sobel_img / sobel_img.max(),cmap='gray')
plt.subplot(222)
plt.title("Add Sobel Mask")
fsbl_img = scaling(img + sobel_img)
show_fsbl = plt.imshow(fsbl_img / fsbl_img.max(),cmap='gray')
plt.show()


plt.figure(figsize=(20,10))
laplace_img = scaling(convolve(img,W = lplc_filter))
show_limg = io.imshow(laplace_img/laplace_img.max())
sl_img = scaling(laplace_img)
plt.subplot(221)
show_limg = plt.imshow(sl_img / sl_img.max(),cmap='gray')
plt.title("Laplacian Filter Applied")
fl_img = scaling(img - sl_img)
plt.subplot(222)
show_limg = plt.imshow(fl_img / fl_img.max(),cmap='gray')
plt.title("Add Laplacian Mask")
plt.show()


plt.figure(figsize=(20,10))
gauss = np.random.normal(128,70,size=ishape[0]*ishape[1]).reshape(ishape[0],ishape[1])
noisy_img = scaling(img + gauss)
plt.subplot(221)
show_nimg = plt.imshow(noisy_img/ noisy_img.max(),cmap='gray')
plt.title("Gaussian Noise Added")
gauss_img = convolve(noisy_img,type = 'nl')
plt.subplot(222)
plt.title("Median Filter Applied")
show_gsimg = plt.imshow(gauss_img / gauss_img.max(),cmap='gray')
plt.show()


gmask = img - low_img
fimg = img + C * gmask
fimg = scaling(fimg)
show_fimg = plt.imshow(fimg/fimg.max(),cmap='gray')
plt.title("Unsharp Masking")
plt.show()
