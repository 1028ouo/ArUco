import cv2
import cv2.aruco as aruco
import numpy as np
import os

# 載入主要影片
cap = cv2.VideoCapture('assets/ArUco_marker.mp4')
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 1.5)
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 1.5)

# 設定輸出影片
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或使用 'XVID'
output_path = 'output_ar_video.mp4'
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_path, fourcc, fps, (frameWidth, frameHeight))

# 設定ArUco偵測器
arucoParams = aruco.DetectorParameters()
arucoDict = aruco.getPredefinedDictionary(aruco.DICT_7X7_50)
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

# 載入要顯示的六個影片
ar_videos = {}
video_files = [
    "assets/video1.mp4", "assets/video2.mp4", "assets/video3.mp4", 
    "assets/video4.mp4", "assets/video5.mp4", "assets/video6.mp4"
]

# 如果沒有實際影片檔案，我們可以生成彩色畫面作為替代
for i in range(6):
    video_path = video_files[i]
    if os.path.exists(video_path):
        ar_videos[i] = cv2.VideoCapture(video_path)
    else:
        # 如果影片不存在，建立彩色畫面當作替代影片
        ar_videos[i] = None

# 定義增強現實處理函數
def augment_marker(frame, corners, marker_id):
    # 取得標記的四個角點
    marker_corners = corners[0][0].astype(np.int32)

    # 從對應影片中讀取一幀
    ret, ar_frame = ar_videos[marker_id].read()
    if not ret:
        ar_videos[marker_id].set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, ar_frame = ar_videos[marker_id].read()
    
    h, w = ar_frame.shape[:2]
    
    ar_corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
    transform_matrix = cv2.getPerspectiveTransform(ar_corners, marker_corners.astype(np.float32))
    warped_ar = cv2.warpPerspective(ar_frame, transform_matrix, (frame.shape[1], frame.shape[0]), 
                                    flags=cv2.INTER_CUBIC)
    
    # 創建遮罩來確定在哪裡置換原始圖像
    mask = np.zeros(frame.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, marker_corners, (255, 255, 255))
    mask = mask / 255.0
    frame_with_ar = frame * (1 - mask) + warped_ar * mask
    
    return frame_with_ar.astype(np.uint8)

print('height {}, width {}'.format(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

# 獲取總幀數用於計算進度
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # 更新當前幀計數和進度
    current_frame += 1
    progress = (current_frame / total_frames) * 100
    
    # 顯示進度條 (50個字符長度)
    bar_length = 50
    filled_length = int(bar_length * current_frame // total_frames)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    # 輸出進度條 (\r讓下一次進度更新覆蓋此行)
    print(f'\r處理進度: |{bar}| {progress:.2f}% ({current_frame}/{total_frames})', end='')
    
    # 調整影像大小以便處理 - 確保使用整數
    frame = cv2.resize(frame, (frameWidth, frameHeight))
    
    # 偵測ArUco標記
    (corners, ids, rejected) = detector.detectMarkers(frame)
    
    # 如果偵測到標記
    if len(corners) > 0:
        # 將ids轉換為一維數組
        if ids is not None:
            ids = ids.flatten()
            
            # 處理每一個被偵測到的標記
            for i in range(len(ids)):
                marker_id = ids[i] % 6  # 確保ID在0-5之間
                marker_corners = [corners[i]]
                frame = augment_marker(frame, marker_corners, marker_id)
    
    # 寫入影片幀到輸出檔案
    out.write(frame)
    
    # 按任意鍵退出
    if cv2.waitKey(20) != -1:
        break

# 完成時換行
print("\n")

# 釋放資源
cv2.destroyAllWindows()
cap.release()
out.release()  # 釋放影片輸出資源
for video in ar_videos.values():
    if video is not None:
        video.release()

print(f'影片已輸出至: {output_path}')
