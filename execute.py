import sys

import matplotlib.pyplot as plt

from algorithmInpainting import *
from biharmonicInpainting import *
import os
from skimage import color, io
import tkinter as tk
from tkinter import filedialog as fd
import cv2
from threading import Thread


def mse(imageA, imageB, mask=None):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images
    if mask == 'spiral':
        imageA = circleMaskImage(imageA)
        imageB = circleMaskImage(imageB)

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar" the two images are
    return err


def pass2D(img):
    if len(img.shape) > 2:
        return color.rgb2gray(img)
    else:
        return img


class Inpaint:
    def __init__(self, fullFilePath: str, resize: float = 0):
        self.image = cv2.imread(fullFilePath)
        if resize:
            self.image = cv2.resize(self.image, (int(self.image.shape[0] * resize), int(self.image.shape[1] * resize)))
        self.fileName = os.path.splitext(os.path.basename(fullFilePath))[0]
        self.filePath = os.path.dirname(fullFilePath)

    def inpaint(self, percent: int, iType: str, masktype: str, show=False):
        if iType == 'algorithm':
            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            imageClass = Algorithm(image, percentInpaint=percent, imgName=self.fileName)
        elif iType == 'biharmonic':
            imageClass = Biharmonic(self.image, percentInpaint=percent, imgName=self.fileName)
        else:
            sys.exit("type given is not a valid option.")

        if masktype == 'random':
            inpainted_img = imageClass.randomInpaint(show=show)
        elif masktype == 'spiral':
            inpainted_img = imageClass.spiralInpaint(show=show)
        else:
            sys.exit("mask given is not a valid option.")

        meanSquardError = mse(pass2D(self.image), pass2D(inpainted_img), mask=masktype)
        # print(percent, '%,', iType, ',', masktype, ',', meanSquardError)
        return meanSquardError


class GUI:
    @staticmethod
    def select_files():
        root = tk.Tk()
        paths = fd.askopenfilenames(filetypes=[('all files', '.*'),
                                               ('text files', '.txt'),
                                               ('image files', '.png'),
                                               ('image files', '.jpg')], title="Choose file(s).")
        paths = root.tk.splitlist(paths)
        root.destroy()
        return paths


class InpaintThread(Thread):
    def __init__(self, img: Inpaint, percent: float, iType: str, masktype: str):
        Thread.__init__(self)

        # set a default value
        self.img = img
        self.percent = percent
        self.iType = iType
        self.masktype = masktype
        self.mse = 0

    def run(self):
        self.mse = self.img.inpaint(self.percent, self.iType, self.masktype, show=False)

    def waitForCompletionAndPrintResult(self):
        # Wait for the thread to finish.
        self.join()
        print(self.img.fileName + ":", self.percent, '%,', self.iType, ',', self.masktype, ',', self.mse)

        name =  str(self.img.fileName) + '_' + str(self.percent) + '_' + str(self.iType) + '_' + str(self.masktype)
        try:
            with open('C:/Users/shawn/Downloads/CCEM data/INPAINTING/Outputs/'+ str(name)+'.txt', 'w') as f:
                f.write(str(self.percent) + '% , ' + str(self.iType) + ' , ' + str(self.masktype) + ' , ' + str(self.mse))
        except:
            print('did not output txt')


if __name__ == '__main__':
    # Import file with tkinter selection.
    fullFilePaths = GUI.select_files()

    for file in fullFilePaths:
        print(file)
        img = Inpaint(file, resize=1)

        threads = []

        percents = [20, 50, 80]
        algTypes = ['algorithm', 'biharmonic']
        maskTypes = ['random', 'spiral']


        for percent in percents:
            for algType in algTypes:
                for maskType in maskTypes:
                    threads.append(InpaintThread(img, percent, algType, maskType))
                    threads[-1].start()
                    #threads[-1].waitForCompletionAndPrintResult()

        for thread in threads:
            thread.waitForCompletionAndPrintResult()
