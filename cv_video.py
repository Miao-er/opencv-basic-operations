import cv2

cap = cv2.VideoCapture("video_tree.mp4")
print(cap.get(5)) #帧率
while True:
    success,img = cap.read() 
    cv2.imshow("Video",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  #waitkey返回按键值
        break