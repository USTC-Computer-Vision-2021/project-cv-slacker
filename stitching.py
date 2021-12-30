import numpy as np
import cv2
from cv2 import Stitcher

if __name__ == "__main__":
 img1 = cv2.imread('1.jpg')
 img2 = cv2.imread('2.jpg')
#  stitcher = cv2.createStitcher(False)
 stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA) # 根据不同的OpenCV版本来调用
 (_result, pano) = stitcher.stitch((img1, img2))
 cv2.imwrite('pano.jpg',pano)
 cv2.waitKey(0)