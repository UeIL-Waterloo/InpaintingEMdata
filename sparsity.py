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
import math
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

def saveAllFigs(params, img, mask, image_defect, image_result):
    cv2.imwrite('C:/Users/shawn/Downloads/CCEM data/INPAINTING/Outputs/original_' + str(params) + '.png', img)
    #cv2.imwrite('C:/Users/shawn/Downloads/CCEM data/INPAINTING/Outputs/mask_' + str(params) + '.png', mask)
    cv2.imwrite('C:/Users/shawn/Downloads/CCEM data/INPAINTING/Outputs/image_defect_' + str(params) + '.png', image_defect)
    cv2.imwrite('C:/Users/shawn/Downloads/CCEM data/INPAINTING/Outputs/image_result_' + str(params) + '.png', image_result)
def circleMaskImage(image):
    blank = np.zeros(image.shape)
    circled = cv2.circle(blank, (blank.shape[0] // 2, blank.shape[1] // 2), blank.shape[0] // 2, color=(255, 255, 255),
                     thickness=-1)
    binary = np.array(circled / circled.max(), dtype=np.uint8)
    return image * binary

def showInpainting(img, mask, image_defect, image_result, name='0'):
    fig, axes = plt.subplots(ncols=2, nrows=2)
    ax = axes.ravel()

    ax[0].set_title('Original image')
    ax[0].imshow(img, cmap=plt.cm.gray)

    ax[1].set_title('Mask')
    ax[1].imshow(mask, cmap=plt.cm.gray)

    ax[2].set_title('Defected image')
    ax[2].imshow(image_defect, cmap=plt.cm.gray)

    ax[3].set_title('Inpainted image')
    ax[3].imshow(image_result, cmap=plt.cm.gray)

    for a in ax:
        a.axis('off')

    fig.tight_layout()
    if name != '0':
        plt.savefig('Images/' + str(name) + '.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

def checkYXduplicates(x,y):
    test = {}
    duplicates = 0
    for i in range(0, len(x)):
        if x[i] in test.keys():
            if y[i] in test[x[i]]:
                duplicates = duplicates + 1
            else:
                test[x[i]].append(y[i])
        else:
            test[x[i]] = [y[i]]
    # print('duplicates: ' + str(duplicates))
    unique_points = sum(len(v) for v in test.values())
    # print('number of unique points: ' + str(unique_points))
    return unique_points, duplicates

def checkXYtupDuplicates(x,y):
    tuples = []
    for i in range(0, len(x)):
        tuples.append((x[i], y[i]))
    uniques = list(set([i for i in tuples]))
    unique_points = len(uniques)
    duplicate_points = len(tuples) - len(uniques)
    newx = []
    newy = []
    for xval, yval in uniques:
        newx.append(xval)
        newy.append(yval)
    return unique_points, duplicate_points, [newx, newy]


def cropImage(img, x1, y1, x2, y2):  # upper left and bottom right
    return img[y1:y2, x1:x2]

class randomSparsity:

    @staticmethod
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

        allPixels = np.arange(totPixels)
        np.random.shuffle(allPixels)
        flaggedPixels = allPixels[:numPixels]

        # Create dict of linearised pixels selected for = 1.
        # flaggedPixels = {}
        #while len(flaggedPixels) < numPixels:
        #    pixel = np.random.randint(0, totPixels)
        #    if pixel not in flaggedPixels.keys():
        #        flaggedPixels[pixel] = 1

        out = [(pixel // Ypixelrange, pixel - (pixel // Ypixelrange * Xpixelrange)) for pixel in flaggedPixels]
        return out

    @staticmethod
    def getRandomMask(img, fracPixels, format='algorithm'):
        if format == 'algorithm':
            mask = np.zeros(img.shape, dtype=np.uint8) # cv2 requires same dim array.
            pixelList = randomSparsity.flagRandomPixelsforInpainting(img, fracPixels)
            for (x, y) in pixelList:
                mask[x, y] = 1
        elif format == 'biharmonic':
            mask = np.zeros(img.shape[:-1], dtype=bool)
            pixelList = randomSparsity.flagRandomPixelsforInpainting(img, fracPixels)
            for (x, y) in pixelList:
                mask[x, y] = True
        else:
            sys.exit("Provide a valid format to getRandomMask.")
        # print("Mask made successfully!")
        return mask

class spiralSparsity:

    @staticmethod
    def Fermat(frequency, totalPoints):
        x = []
        y = []
        for i in range(0, totalPoints):
            x.append(math.sqrt(i) * math.cos(frequency * i))
            y.append(math.sqrt(i) * math.sin(frequency * i))
        return x, y

    @staticmethod
    def CLV(t, length, frequency = 1):
        x = []
        y = []
        for i in range(0, t):
            sq = np.sqrt(i)
            x.append(int(sq * np.cos(frequency * sq)))
            y.append(int(sq * np.sin(frequency * sq)))

        x = np.interp(x, [min(x), max(x)], [0, length - 1]).astype(int)
        y = np.interp(y, [min(y), max(y)], [0, length - 1]).astype(int)
        return x, y
        # mask = np.ones((width, height),  dtype=maskType)
        # for i in range(totalPoints):
        #     sq = np.sqrt(i)
        #     x = min(int(sq * np.cos(frequency * sq)) + shiftAmount, width - 1)
        #     y = min(int(sq * np.sin(frequency * sq)) + shiftAmount, height - 1)
        #     mask[x][y] = 0
        # return mask

    @staticmethod
    def CLVmask(img, percentInpaint, format='algorithm'):
        if format == 'algorithm':
            width, height = img.shape
        if format == 'biharmonic':
            width, height = img.shape[:-1]
        # Crop image if the width and height are not equal.
        if width != height:
            assert(False)
            print("Image width and height are not equal, cropping in top left to largest square.")
            length = min(width, height)
            img = cropImage(img, 0, 0, length, length)
            width, height = img.shape
            # plt.imshow(img, cmap='Greys_r', interpolation='nearest')
            # plt.axis('off')
            # plt.show()
        circleArea = math.pi * (width / 2) ** 2

        percentRemoved = 100 - percentInpaint
        # Dose will be measured as a fraction of the total pixel points.
        # Though some pixels are sampled more than once, the dose should be evenly distributed throughout the sample.
        # i.e. dose on the sample is linear but measurements are not even across all detector points.
        if percentRemoved == 20:
            t = int(math.pi * (width / 2) ** 2 / 2.6)
        elif percentRemoved == 50:
            t = int(math.pi * (width / 2) ** 2 / 0.46)
        elif percentRemoved == 80:
            t = int(math.pi * (width / 2) ** 2 / 0.19)
        else:
            assert(False)

        x, y = spiralSparsity.CLV(t, length = width)


        unique_points, duplicate_points = checkYXduplicates(x, y)
        percent_inpainted = round(unique_points/circleArea*100)
        # print(percent_inpainted)
        # TODO scans should not have gaps in path, update for more representative result.

        def shift(l, shiftval):
            return l
            #return [i+shiftval for i in l]
        shiftx = shift(x, width//2)
        shifty = shift(y, width//2)

        if format == 'algorithm':
            mask = np.ones(img.shape, dtype=np.uint8) # cv2 requires same dim array.
            for i in range(len(shiftx)):
                mask[shiftx[i], shifty[i]] = 0
        elif format == 'biharmonic':
            mask = np.ones(img.shape[:-1], dtype=bool)
            for i in range(len(shiftx)):
                mask[shiftx[i]][shifty[i]] = False

        else:
            sys.exit("Provide a valid format to getRandomMask.")

        # print("Mask made successfully!")


        return mask, percent_inpainted
