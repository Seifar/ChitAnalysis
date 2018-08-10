import cv2
import glob, os
import numpy as np
import imutils
from PIL import Image
import pytesseract
import re
import time
import CSVHelper
import multiprocessing
from multiprocessing.pool import ThreadPool
import concurrent.futures as cf

DEBUG = False

def showIMG(mat, name, delay=0):
    if not DEBUG:
        return
    cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name, mat)
    cv2.resizeWindow(name, 1000, 1000)
    cv2.waitKey(delay)
    cv2.destroyWindow(name)


def splitChit(original):
    # edge detection
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    # showIMG(edged, "Canny")
    edged = cv2.GaussianBlur(edged, (3, 3), 0)

    # finding contours
    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = filter(lambda x: original.shape[0] * original.shape[1] / 6 < cv2.contourArea(x), cnts)
    screenCnt = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt.append(approx)
    # show contours
    # contourImage = original.copy()
    # cv2.drawContours(contourImage, screenCnt, -1, (0, 255, 0), 5)
    # showIMG(contourImage, "contours")

    # create individual Images
    chits = []
    for c in screenCnt:
        pts1 = [c[0][0], c[1][0], c[2][0], c[3][0]]
        # sort the contour points in the right way
        pts1 = sorted(pts1, key=lambda x: x[1])
        if pts1[0][0] > pts1[1][0]:
            pts1[0], pts1[1] = pts1[1], pts1[0]
        if pts1[2][0] > pts1[3][0]:
            pts1[2], pts1[3] = pts1[3], pts1[2]
        # transform
        pts1 = np.float32(pts1)
        pts2 = np.float32([[0, 0], [2000, 0], [0, 1000], [2000, 1000]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(original, M, (2000, 1000))
        chits.append(dst)
    return chits


# iput: single chit in landscape orientation (top either on the left or right)
# returns: chit in portrait mode
def rotateChit(chit):
    rows, cols, _ = chit.shape
    grey = cv2.cvtColor(chit, cv2.COLOR_BGR2GRAY)
    leftfill = 0
    for c in range(cols):
        for r in range(round(rows * 0.05)):
            leftfill += grey[r][c]

    rightfill = 0
    for c in range(cols):
        for r in range(round(rows * 0.95), rows):
            leftfill += grey[r][c]

    return imutils.rotate_bound(chit, 90 if leftfill < rightfill else 270)


def getNumber(chit):
    tmp = chit[300:370, 90:270]
    cv2.threshold(tmp, 127, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(chit[300:370, 90:270], config="-c tessedit_char_whitelist=ABCD0123456789")
    if not re.compile("[0-9][0-9]?[A-D]").match(text):
        return None
    if len(text) == 2:
        text = "0" + text
    return text


#TODO rewrite with Tensorflow
def getPoints(chit):
    text = pytesseract.image_to_string(chit[1330:1490, 800:999], config="-c tessedit_char_whitelist=0123456789")
    text = text.replace(" ", "")
    text = text.replace("\n", "")
    if text == "":
        return -1
    return int(text)

# takes a whole chit file and returns target number, score, 10 and x of
def analyseAll(file):
    # display original file
    original = cv2.imread("" + file)
    showIMG(original, "original")

    chits = splitChit(original)

    chits = map(lambda x: rotateChit(x), chits)
    ret = []
    # show them
    for chit in chits:
        showIMG(chit, "chit")
        number = getNumber(chit)
        points = getPoints(chit)
        if number == None:
            ret.append(None)
            continue
        print([number, points])
        ret.append({CSVHelper.TARGET: number, CSVHelper.SCORE: points, CSVHelper.TEN: 0, CSVHelper.X: 0})
    return ret

# MAIN PROGRAMM
startDir = os.getcwd()
os.chdir("data")
dataset = []
print("Started Running...")
timer = time.time()
fileIterator = glob.glob("*.[Jj][Pp][Gg]")
total = len(fileIterator)

# start analysation in a separate process for each chut
processPool = cf.ProcessPoolExecutor(max_workers=1 if DEBUG else multiprocessing.cpu_count() * 5)
for i in fileIterator:
    dataset.append(processPool.submit(analyseAll, i))
cf.wait(dataset)

#unpack data received from pool
dataset = map(lambda  date: date.result(), dataset)
dataset = [item for sublist in dataset for item in sublist] #flatten

#cleanup dataset & write to csv
oldsize = len(dataset)
dataset = filter(lambda date:date is not None, dataset)
dataset = sorted(dataset, key=lambda date: date[CSVHelper.TARGET])
os.chdir(startDir)
CSVHelper.Writer("score.csv").writeAll(dataset)


timer = time.time() - timer
print("Recognized:" + str(len(dataset)))
print("Number failed: " + str(oldsize - len(dataset)) +" of " + str(oldsize) + " found chits, " + str(total*4) + " total")
print("This took %d seconds" % timer)

