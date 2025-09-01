import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import os

arucoDict = aruco.getPredefinedDictionary(aruco.DICT_7X7_50)

plt.figure(figsize=(10, 15))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

for i in range(0,6):
    plt.subplot(3, 2, 1+i)  
    marker_image = aruco.generateImageMarker(arucoDict,i,200)
    plt.imshow(marker_image,cmap='gray')
    plt.axis(False)

plt.savefig('aruco_markers.png', dpi=300, bbox_inches='tight', pad_inches=1)