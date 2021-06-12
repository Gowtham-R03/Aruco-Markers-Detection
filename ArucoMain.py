import cv2
import cv2.aruco as aruco
import numpy as np
import os

def loadAugmentImages(path):
    myList = os.listdir(path)
    noOfMarkers = len(myList)
    #print("Total no of markers detected is:",noOfMarkers)
    augDictionary = {}
    for imgPath in myList:
        key = int(os.path.splitext(imgPath)[0])  #splitting 23 and png
        imgAug = cv2.imread(f'{path}/{imgPath}')
        augDictionary[key] = imgAug  #23: imgae23, 40:image40
    return  augDictionary


#markerSize is (6x6), draw whn the parameters satisfied
def findArucoMarkers(img, markerSSize=6, totalMarkers=250, draw=True):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #Prepocessing
    key = getattr(aruco,f'DICT_{markerSSize}X{markerSSize}_{totalMarkers}')
    arucoDictionary = aruco.Dictionary_get(key)
    arucoParameters = aruco.DetectorParameters_create()
    bb0x, ids, rejectedIds =aruco.detectMarkers(imgGray,arucoDictionary,parameters=arucoParameters)
    #print(ids)
    if draw:
        aruco.drawDetectedMarkers(img,bb0x)
    return  [bb0x, ids]

#for Augumentation
def arucoAugment(bbox, id, img, imgAug, drawId = True):
    #four corner points
    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]

    h, w, c = imgAug.shape

    #wrap prespective
    pts1 = np.array([tl,tr,br,bl])
    pts2 = np.float32([[0,0],[w,0],[w,h],[0,h]])
    matrix, _ = cv2.findHomography(pts2, pts1)
    imgoutput = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0])) #except aruco all will black
    cv2.fillConvexPoly(img, pts1.astype(int), (0,0,0)) #aruco will black
    imgoutput = img + imgoutput #combing above two we can get image without black

    if drawId:
        cv2.putText(imgoutput, str(id), tl, cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)

    return imgoutput




def main():
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)

    while True:
        success, img = cap.read()
        augDics = loadAugmentImages("Aruco Augment")
        arucoFound = findArucoMarkers(img)

        #Looping of augment images on markers one by one
        if len(arucoFound[0])!=0:
            for bbox, id in zip(arucoFound[0],arucoFound[1]):
                if int(id) in augDics.keys():
                    img = arucoAugment(bbox, id,img, augDics[int(id)])

        cv2.imshow("Camera View",img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
