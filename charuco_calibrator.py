import cv2
import cv2.aruco as aruco
import numpy as np

cap = cv2.VideoCapture('assets/ChArUco_board.mp4')
# 原始畫面有點大，為了有利於顯示這份講義所以縮小。    
totalFrame   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
frameHeight  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2

arucoParams = aruco.DetectorParameters()
arucoParams.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
arucoDict   = aruco.getPredefinedDictionary(aruco.DICT_6X6_250) 

# Create the ArUco detector
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

# 必須描述ChArUco board的尺寸規格
gridX        = 5  # 水平方向5格
gridY        = 7  # 垂直方向7格
squareSize   = 4  # 每格為4cmX4cm

# ArUco marker為2cmX2cm
charucoBoard = aruco.CharucoBoard((gridX, gridY), squareSize, squareSize / 2, arucoDict)

# 初始化相機矩陣與畸變係數
cameraMatrixInit = np.array([[1000., 0., frameWidth / 2.],
                             [0., 1000., frameHeight / 2.],
                             [0., 0., 1.]])
distCoeffsInit = np.zeros((5, 1))

charucoParams = aruco.CharucoParameters()
charucoParams.tryRefineMarkers = True
charucoParams.cameraMatrix = cameraMatrixInit
charucoParams.distCoeffs = distCoeffsInit
charucoDetector = aruco.CharucoDetector(charucoBoard, charucoParams, arucoParams)

print('height {}, width {}'.format(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
refinedStrategy = True
criteria        = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
frameId        = 0
objPointsList  = []
imgPointsList  = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.resize(frame, (frameWidth, frameHeight)) 
    corners, ids, _, _ = charucoDetector.detectBoard(frame)

    if corners is not None and corners.shape[0] > 0:
        aruco.drawDetectedCornersCharuco(frame, corners, ids)
        objPoints, imgPoints = charucoBoard.matchImagePoints(corners, ids)
    
        if frameId % 100 == 50 and objPoints.shape[0] >= 4: 
            objPointsList.append(objPoints)
            imgPointsList.append(imgPoints)

    cv2.imshow('Analysis of a CharUco board for camera calibration', frame)
    if cv2.waitKey(20) != -1:
        break
        
    frameId += 1

cv2.destroyAllWindows()
cap.release()

ret, charuco_cameraMatrix, charuco_distCoeffs, aruco_rvects, aruco_tvects = cv2.calibrateCamera(
    objPointsList, imgPointsList, (frameWidth, frameHeight), cameraMatrixInit, distCoeffsInit
)
print("Camera Matrix:\n", charuco_cameraMatrix)
print("Distortion Coefficients:\n", charuco_distCoeffs)