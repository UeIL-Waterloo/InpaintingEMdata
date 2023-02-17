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
    print('duplicates: ' + str(duplicates))
    unique_points = sum(len(v) for v in test.values())
    print('number of unique points: ' + str(unique_points))
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

    @staticmethod
    def getRandomMask(img, fracPixels):
        mask = np.zeros(img.shape, dtype=np.uint8)
        pixelList = randomSparsity.flagRandomPixelsforInpainting(img, fracPixels)
        for (x,y) in pixelList:
            mask[x,y] = 1
        print("Mask made successfully!")
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
    def CLV(frequency, totalPoints):
        x = []
        y = []
        for i in range(0, totalPoints):
            xval = int(math.sqrt(i) * math.cos(frequency * math.sqrt(i)))
            yval = int(math.sqrt(i) * math.sin(frequency * math.sqrt(i)))
            x.append(xval)
            y.append(yval)
        return x, y

    @staticmethod
    def CLVmask(img, frequency = 1.16):
        width, height = img.shape
        # Crop image if the width and height are not equal.
        if width != height:
            print("Image width and height are not equal, cropping in top left to largest square.")
            length = min(width, height)
            img = cropImage(img, 0, 0, length, length)
            width, height = img.shape
            # plt.imshow(img, cmap='Greys_r', interpolation='nearest')
            # plt.axis('off')
            # plt.show()
        circleArea = math.pi * (width / 2) ** 2

        t = int(width**2/4)
        # Dose will be measured as a fraction of the total pixel points.
        # Though some pixels are sampled more than once, the dose should be evenly distributed throughout the sample.
        # i.e. dose on the sample is linear but measurements are not even across all detector points.
        x,y = spiralSparsity.CLV(frequency=frequency, totalPoints=t)

        # print('total points: '+str(len(x)))
        # print('dose: ' + str(len(x)/circleArea*100) + '%')
        unique_points, duplicate_points = checkYXduplicates(x, y)
        inpainted = round(100 - unique_points/circleArea*100)
        # print('inpainted: ' + str(inpainted) + '%')
        # TODO scans should not have gaps in path, update for more representative result.

        def shift(l, shiftval):
            return [i+shiftval for i in l]
        shiftx = shift(x, width//2)
        shifty = shift(y, width//2)

        # plt.plot(y,x)
        # plt.plot(shifty, shiftx)
        # plt.show()
        # plt.plot(range(len(x)), x, 'b')
        # plt.plot(range(len(y)), y, 'r')
        # plt.show()

        mask = np.zeros(img.shape, dtype=np.uint8)
        for i in range(len(shiftx)):
            mask[shiftx[i], shifty[i]] = 1
        print("Mask made successfully!")

        plt.imshow(mask, cmap='Greys_r', interpolation='nearest')
        plt.axis('off')
        plt.show()

        return mask, inpainted


# path = 'Images/test_image.tif'
#
# img = cv2.imread(path)
# img = cv2.resize(img, (int(img.shape[0] * 0.5), int(img.shape[1] * 0.5)))
# # img = cropImage(img, 0,0,int(img.shape[0]*0.5), img.shape[0])
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # plt.imshow(img, cmap='Greys_r', interpolation='nearest')
# # plt.axis('off')
# # plt.show()
#
# spiralSparsity.CLVmask(img, 10)