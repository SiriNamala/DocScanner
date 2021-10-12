#Document Scanner
import cv2
import numpy as np
import epydoc

img = cv2.imread("Resources/img3.jpg")
widthImg = 640
heightImg= 480
#preprocessing of image to detect the edges in the image
def pre_processing(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(3,3),1)
    imgCanny = cv2.Canny(imgBlur,100,100)
    kernel=np.ones((5,5))
    imgDilation = cv2.dilate(imgCanny,kernel,iterations=6)
    imgThres = cv2.erode(imgDilation,kernel,iterations=4)
    return imgThres

#contours - find the biggest contour of our image, give threshold for area
def getContours(img):
    big = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >270000:
            #cv2.drawContours(imgContour, cnt,-1,(255,0,0),1)
            peri = cv2.arcLength(cnt,True)
            #print(peri)
            approx = cv2.approxPolyDP(cnt,0.02*peri, True)

            #as it loops, need to find bigger one
            if area >5000 and len(approx) == 4:
                big = approx
                maxArea = area
    cv2.drawContours(imgContour, big, -1, (0, 255, 0), 20)
    print(maxArea)
    return big



def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)
    print("add",add)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff =np.diff(myPoints,axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
#    print("my points",myPointsNew)
    return myPointsNew


def getWarp(img,biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    return imgOutput




imgContour = img.copy()
imgThres = pre_processing(img)
big = getContours(imgThres)
print(big)
imgWarped = getWarp(img,big)
cv2.imshow("Result",imgWarped)
cv2.waitKey(0)








