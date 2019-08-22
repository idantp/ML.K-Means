# Author: Idan Twito
# ID: 311125249
import numpy as np
import scipy.io as sio
import math
import matplotlib.pyplot as plt
from init_centroids import init_centroids

from scipy.misc import imread


class clusterTotalInfo():
    rColorSum = 0
    rAvg = 0
    gColorSum = 0
    gAvg = 0
    bColorSum = 0
    bAvg = 0
    numOfPixels = 0


# Function Name: findClosestCentroid
# Function Input: centroidArr, pixel, centroidNum
# Function Output: closestCentroidIndex
# Function Operation: the function finds and returns the closest centroid to the given pixel.

def findClosestCentroid(centroidArr, pixel, centroidNum):
    minDifference = math.sqrt(math.pow((centroidArr[0][0] - pixel[0]), 2)
                              + math.pow((centroidArr[0][1] - pixel[1]), 2)
                              + math.pow((centroidArr[0][2] - pixel[2]), 2))
    closestCentroidIndex = 0
    for centroidIndex in range(centroidNum):
        differenceFromPixel = math.sqrt(
            math.pow((centroidArr[centroidIndex][0] - pixel[0]), 2) + math.pow(
                (centroidArr[centroidIndex][1] - pixel[1]), 2) + math.pow(
                (centroidArr[centroidIndex][2] - pixel[2]), 2))
        if differenceFromPixel < minDifference:
            minDifference = differenceFromPixel
            closestCentroidIndex = centroidIndex
    return closestCentroidIndex


# Function Name: distanceToClosestCentroid
# Function Input: centroidArr, pixel, centroidNum
# Function Output: closestCentroidIndex
# Function Operation: the function finds and returns the distance to the closest centroid from
#                     the given pixel.

def distanceToClosestCentroid(centroidArr, pixel, centroidNum):
    minDifference = math.sqrt(math.pow((centroidArr[0][0] - pixel[0]), 2)
                              + math.pow((centroidArr[0][1] - pixel[1]), 2)
                              + math.pow((centroidArr[0][2] - pixel[2]), 2))
    for centroidIndex in range(centroidNum):
        differenceFromPixel = math.sqrt(
            math.pow((centroidArr[centroidIndex][0] - pixel[0]), 2) + math.pow(
                (centroidArr[centroidIndex][1] - pixel[1]), 2) + math.pow(
                (centroidArr[centroidIndex][2] - pixel[2]), 2))
        if differenceFromPixel < minDifference:
            minDifference = differenceFromPixel
    return minDifference


def print_cent(cent):
    if type(cent) == list:
        cent = np.asarray(cent)
    if len(cent.shape) == 1:
        return ' '.join(str(np.floor(100 * cent) / 100).split()).replace('[ ', '[').replace('\n',
                                                                                            ' ').replace(
            ' ]', ']').replace(' ', ', ')
    else:
        return ' '.join(str(np.floor(100 * cent) / 100).split()).replace('[ ', '[').replace('\n',
                                                                                            ' ').replace(
            ' ]', ']').replace(' ', ', ')[1:-1]


# Function Name: clustersDistribution
# Function Input: centroidArr, pixelsArr, centroidsNum, loopIndex
# Function Output: clustersInfoArr
# Function Operation: the function assigns each pixel to it's closest centroid, and then
#                     determines the color of each centroid as the avg of its pixels colors.
def clustersDistribution(centroidArr, pixelsArr, centroidsNum, loopIndex):
    clustersInfoArr = []
    for amount in range(centroidsNum):
        clustersInfoArr.append(clusterTotalInfo())
    # assignins each pixel to its closest centroid
    for pixelIndex in range(len(pixelsArr)):
        closestCentroidIndex = findClosestCentroid(centroidArr, pixelsArr[pixelIndex], centroidsNum)
        clustersInfoArr[closestCentroidIndex].rColorSum += pixelsArr[pixelIndex][0]
        clustersInfoArr[closestCentroidIndex].gColorSum += pixelsArr[pixelIndex][1]
        clustersInfoArr[closestCentroidIndex].bColorSum += pixelsArr[pixelIndex][2]
        clustersInfoArr[closestCentroidIndex].numOfPixels += 1
    # calculating the avg color of each centroid
    for index in range(centroidsNum):
        pixelsInCluster = clustersInfoArr[index].numOfPixels
        if pixelsInCluster != 0:
            clustersInfoArr[index].rAvg = (clustersInfoArr[index].rColorSum) / pixelsInCluster
            clustersInfoArr[index].gAvg = (clustersInfoArr[index].gColorSum) / pixelsInCluster
            clustersInfoArr[index].bAvg = (clustersInfoArr[index].bColorSum) / pixelsInCluster
        # printResult(index, loopIndex, clustersInfoArr, centroidsNum)
    return clustersInfoArr


def lossFunc(pixelsArr, centroidArr, centroidsNum):
    distance = 0
    numOfPixels = 0
    for pixel in pixelsArr:
        distance += (distanceToClosestCentroid(centroidArr, pixel, centroidsNum) ** 2)
        numOfPixels += 1
    return (distance / numOfPixels)


def main():
    # array of centroids amount
    centroidsNumArr = [2, 4, 8, 16]
    for centroidsNum in centroidsNumArr:
        # data preperation (loading, normalizing, reshaping)
        path = 'dog.jpeg'
        A = imread(path)
        A_norm = A.astype(float) / 255.
        img_size = A_norm.shape
        X = A_norm.reshape(img_size[0] * img_size[1], img_size[2])
        # initiallizing the centroids according to the amount of centroids
        centroidArr = init_centroids(X, centroidsNum)
        print("k={0}:".format(centroidsNum))
        # 10 iterations for each value of centroids amount
        lossArray = []
        for lossIndex in range(11):
            lossArray.append(0)
        for index in range(11):
            clusterArrInfo = clustersDistribution(centroidArr, X, centroidsNum, index)
            print("iter {0}: ".format(index), end='')
            print(print_cent(centroidArr))
            lossResult = lossFunc(X, centroidArr, centroidsNum)
            lossArray[index] = lossResult
            for i in range(centroidsNum):
                centroidArr[i][0] = clusterArrInfo[i].rAvg
                centroidArr[i][1] = clusterArrInfo[i].gAvg
                centroidArr[i][2] = clusterArrInfo[i].bAvg
        for pixelIndex in range(len(X)):
            X[pixelIndex] = centroidArr[
                findClosestCentroid(centroidArr, X[pixelIndex], centroidsNum)]
        # plt.plot(lossArray)
        # plt.title('k-means test, k=%d' % centroidsNum)
        # plt.ylabel('loss')
        # plt.xlabel('iteration number')
        # plt.show()
        # plot the image
        # plt.imshow(A_norm)
        # plt.grid(False)
        # plt.show()


if __name__ == "__main__":
    main()
