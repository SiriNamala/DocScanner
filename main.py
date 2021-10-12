#Document Scanner
import cv2
import numpy as np

capture = cv2.VideoCapture("Resources/vid0.mp4")
widthImg = 640
heightImg= 480
#preprocessing of image to detect the edges in the image
def pre_processing(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(1,1),1)
    imgCanny = cv2.Canny(imgBlur,100,100)
    kernel=np.ones((1,1))
    imgDilation = cv2.dilate(imgCanny,kernel,iterations=4)
    imgThres = cv2.erode(imgDilation,kernel,iterations=2)
    return imgThres

#contours - find the biggest contour of our image, give threshold for area
def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area >5000:
            #cv2.drawContours(imgContour, cnt,-1,(255,0,0),3)
            peri = cv2.arcLength(cnt,True)
            #print(peri)
            approx = cv2.approxPolyDP(cnt,0.02*peri, True)
            #as it loops, need to find bigger one
            if area >maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    return biggest

def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)
    print("add",add)



def getWarp(img,biggest):
    reorder(biggest)
    print(biggest.shape)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    return imgOutput






while True:
    success, img = capture.read()
    cv2.resize(img,(widthImg,heightImg))
    imgContour = img.copy()
    imgThres = pre_processing(img)
    biggest = getContours(imgThres)
    print(biggest)
    imgWarped = getWarp(img,biggest)
    cv2.imshow("Result",imgWarped)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break












