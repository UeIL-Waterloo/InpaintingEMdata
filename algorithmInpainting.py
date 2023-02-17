# MIT License
#
# Copyright (c) 2023 Nicolette Shaw
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import sys

import matplotlib.pyplot as plt
import cv2
import random
from sparsity import *

# INPAINT_NS = Use Navier-Stokes based method
# INPAINT_TELEA = Use the algorithm proposed by Alexandru Telea [209]

path = 'Images/test_image.png'

img = cv2.imread(path)
img = cv2.resize(img, (int(img.shape[0] * 0.5), int(img.shape[1] * 0.5)))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# img = cv2.resize(img, (100,100))

# Nonzero pixels are to be inpainted with opencv/ white are inpainted.
percentInpainted = 80
a = randomSparsity.getRandomMask(img, fracPixels=percentInpainted)
# a, percentInpainted = spiralSparsity.CLVmask(img)
# a = 1-a

masked = img * (1-a)

# Produce random code in name to label datasets.
name = 'Images/Liposome_random' + str(random.randint(0,1000)) + '_'

plt.imshow(img, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.show()
cv2.imwrite(name+'originalImage.png', img)


plt.imshow(masked, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.show()
cv2.imwrite(name+'mask'+str(percentInpainted)+'.png', masked)


dst = cv2.inpaint(img,a,1,cv2.INPAINT_TELEA)
plt.axis('off')
plt.imshow(dst, cmap='Greys_r', interpolation='nearest')
plt.show()
cv2.imwrite(name+'imageInpainted'+str(percentInpainted)+'.png', dst)


dst = cv2.inpaint(masked,a,1,cv2.INPAINT_TELEA)
plt.axis('off')
plt.imshow(dst, cmap='Greys_r', interpolation='nearest')
plt.show()
cv2.imwrite(name+'maskInpainted'+str(percentInpainted)+'.png', dst)
