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
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.restoration import inpaint
from sparsity import *
from skimage import img_as_ubyte

class Biharmonic:
    def __init__(self, img, percentInpaint, imgName):
        self.img = img
        self.percentInpaint = percentInpaint
        self.imgName = imgName


    def randomInpaint(self, show=True):
        mask = randomSparsity.getRandomMask(self.img, fracPixels=self.percentInpaint, format='biharmonic')
        image_defect = self.img * ~mask[..., np.newaxis]
        image_result = inpaint.inpaint_biharmonic(image_defect, mask, channel_axis=-1)

        image_resultgray = img_as_ubyte(image_result)

        saveAllFigs(self.imgName + '_biharmonic_random_' + str(self.percentInpaint), self.img, mask, image_defect, image_resultgray)

        if show:
            showInpainting(self.img, mask, image_defect, image_result, name='random_biharmonic')

        return image_result

    def spiralInpaint(self, show=True):
        mask, percentInpainted = spiralSparsity.CLVmask(self.img, percentInpaint=self.percentInpaint, format='biharmonic')
        image_defect = self.img * np.invert(mask)[..., np.newaxis]
        image_result = inpaint.inpaint_biharmonic(image_defect, mask, channel_axis=-1)

        image_resultgray = img_as_ubyte(image_result)

        saveAllFigs(self.imgName + '_biharmonic_spiral_' + str(self.percentInpaint), self.img, mask, image_defect, image_resultgray)

        if show:
            showInpainting(self.img, mask, image_defect, circleMaskImage(image_result), name='spiral_biharmonic')

        return image_result
