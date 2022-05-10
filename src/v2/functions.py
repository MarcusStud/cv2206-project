import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import cv2
from scipy.stats import stats
from PIL import Image
from PIL import ImageFilter
import random
import os
import platform

def getIm(typ, amount, arr):
    if(platform.system() == "Windows"):
      Negative = os.path.abspath("testdata")
      Positive = os.path.abspath("testdata")
      Negative = Negative.replace("\src\dist\main", "", 1)
      Positive = Positive.replace("\src\dist\main", "", 1)
      Negative = Negative + "\\Negative\\"
      Positive = Positive + "\\Positive\\"
      
    else:
      Positive = "cv2206-project/testdata/Positive/"
      Negative = "cv2206-project/testdata/Negative/"
    dupes = []
    for _ in range (amount):
        getint, dupes = getUniqueRand(dupes)
        get = str(getint)
        digits = len(get)
        if digits == 5 and getint < 19379 and typ:
            get = get + "_1"
        while digits < 5:
            get = "0" + get
            digits = len(get)
        if typ:
            get = Positive + get + ".jpg"
        else:
            get = Negative + get + ".jpg"
        im = cv2.imread(get)
        arr.append(im)

#Returns a rendom number that has not yet been added to the given array
def getUniqueRand(dupes):
    dupe = True
    while dupe:
        ret = random.randint(1, 1000)
        dupe = ret in dupes
    dupes.append(ret)
    return ret, dupes

# Returns two arrays positive and negative which contains amount images each
def getRandomImages(amount, Pos, Neg):
    getIm(True, amount, Pos)
    getIm(False, amount, Neg)
    return Pos, Neg

def contours(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_lencnt = max([len(x) for x in contours])
        return max_lencnt
    return 0

def results(images, sampleType):
  positive = 0
  negative = 0
  for i in range(len(images)):
    cont = contours(images[i])
    if cont > 50:
      positive += 1
    else:
      negative += 1
  if sampleType.lower() == "positive":
    txt1 = ("Results after testing", len(images), "images that are positive of having cracks:")
    txt2 = ("Total Positives: ", positive, "/", len(images), "")
    txt3 = ("Total False-negatives: ", negative, "/", len(images))
    txt4 = ("Accuracy:" , (positive/len(images))*100, "%")
    list = [txt1,txt2,txt3,txt4]
    return list
 
  if sampleType.lower() == "negative":
    txt1 = ("Results after testing", len(images), "images that are negative of having cracks:")
    txt2 = ("Total Negatives: ", negative, "/", len(images))
    txt3 = ("Total False-positives: ", positive, "/", len(images))
    txt4 = ("Accuracy:" , (negative/len(images))*100,)
    list = [txt1,txt2,txt3,txt4]
    return list

def checkCrackSelectedImage(img):
  cont = contours(img)
  if cont > 50:
   return True
  else:
   return False
