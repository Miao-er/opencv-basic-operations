import numpy as np
import cv2 as cv
cv.namedWindow("images")
def nothing():
    pass
cv.createTrackbar("s1","images",0,255,nothing)
cv.createTrackbar("s2","images",0,255,nothing)
img = cv.imread("lena.jpg",0)
while(1):
    s1 = cv.getTrackbarPos("s1","images")
    s2 = cv.getTrackbarPos("s2","images")
    out_img = cv.Canny(img,s1,s2)
    cv.imshow("img",out_img)
    k = cv.waitKey(1)
    if k==ord("q"):
        break
cv.destroyAllWindows()