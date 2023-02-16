import matplotlib.pyplot as plt
import cv2
import numpy as np
import random

# INPAINT_NS = Use Navier-Stokes based method
# INPAINT_TELEA = Use the algorithm proposed by Alexandru Telea [209]


def flagRandomPixelsforInpainting(img, fracPixels):
    '''

    :param img:
    :param fracPixels: percentage (0-100)
    :return:
    '''
    Xpixelrange = img.shape[0]
    Ypixelrange = img.shape[1]
    totPixels = Xpixelrange * Ypixelrange
    numPixels = int(totPixels * fracPixels/100.0)
    flaggedPixels = {}

    # Create dict of linearised pixels selected for = 1.
    while len(flaggedPixels) < numPixels:
        pixel = np.random.randint(0, totPixels)
        if pixel not in flaggedPixels.keys():
            flaggedPixels[pixel] = 1

    # Check if it is working.
    # for i in range(0,Xpixelrange * 50):
    #     flaggedPixels[i] = 1

    # Delinearise pixels into coordinates.
    # x = pixel // Ypixelrange
    # y = pixel - (x * Xpixelrange)
    return [(pixel // Ypixelrange, pixel - (pixel // Ypixelrange * Xpixelrange)) for pixel in flaggedPixels]

def getRandomMask(img, fracPixels):
    mask = np.zeros(img.shape, dtype=np.uint8)
    pixelList = flagRandomPixelsforInpainting(img, fracPixels)
    for (x,y) in pixelList:
        mask[x,y] = 1
    print("Mask made successfully!")
    return mask

path = 'Images/test_image.tif'

img = cv2.imread(path)
img = cv2.resize(img, (int(img.shape[0] * 0.5), int(img.shape[1] * 0.5)))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# img = cv2.resize(img, (100,100))

# Nonzero pixels are to be inpainted with opencv/ white are inpainted.
percentInpainted = 50
a = getRandomMask(img, percentInpainted)

masked = img * (1-a)

# Produce random code in name to label datasets.
name = 'Images/Liposome_' + str(random.randint(0,1000)) + '_'

plt.imshow(img, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
# plt.savefig(name+'originalImage.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()

plt.imshow(masked, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
# plt.savefig(name+'mask'+str(percentInpainted)+'.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()

dst = cv2.inpaint(img,a,1,cv2.INPAINT_TELEA)
plt.axis('off')
plt.imshow(dst, cmap='Greys_r', interpolation='nearest')
# plt.savefig(name+'imageInpainted'+str(percentInpainted)+'.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()

dst = cv2.inpaint(masked,a,1,cv2.INPAINT_TELEA)
plt.axis('off')
plt.imshow(dst, cmap='Greys_r', interpolation='nearest')
# plt.savefig(name+'maskInpainted'+str(percentInpainted)+'.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()