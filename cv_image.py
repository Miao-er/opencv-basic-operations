import cv2
import numpy as np
#导入图片
img = cv2.imread("lena.jpg")
# cv2.imshow("output",img)
# cv2.waitKey(0)  #该参数为零表示无限延迟，此处为1000ms

#转化为灰度图像
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray image", imgGray)
#cv2.waitKey(0)

#模糊图像
imgBlur = cv2.GaussianBlur(img,(5,5),0)
# cv2.imshow("blur image",imgBlur)
# cv2.waitKey(0)

#边缘检测Canny算法
low = 100
high = 200
imgCanny = cv2.Canny(img,low,high)  #高低阈值
#cv2.imshow("canny image",imgCanny)
#cv2.waitKey(0)

#形态学操作（针对二值图
kernel = np.ones((3,3))
#膨胀，卷积核平移，对应区域有any 1，则变为1，扩大白色区域
imgDilation = cv2.dilate(imgCanny,kernel,iterations= 1) #迭代次数
#cv2.imshow("dilate image",imgDilation)
#cv2.waitKey(0)
#膨胀，卷积核平移，对应区域有all 1，则变为1，缩小白色区域
imgEroded = cv2.erode(imgDilation,kernel,iterations= 2)
# cv2.imshow("erode image", imgEroded)
# cv2.waitKey(0)

#改变大小
#shape先高度后宽度(y,x),resize先宽度后高度(x,y)
imgResize = cv2.resize(img,(300,300))
print(img.shape,imgResize.shape)
#cv2.imshow("Resized image",imgResize)
#cv2.waitKey(0)

#裁剪
imgCropped = imgResize[0:200,100:300]#先高后宽
# cv2.imshow("Cropped image",imgCropped)
# cv2.waitKey(0)

#绘图
#RGB顺序:B-G-R
img_empty = np.zeros((512,512,3),np.uint8) #创建空图
img_empty[:] = 255,0,0                     #底色为蓝色
#坐标值先宽后高
cv2.line(img_empty,(0,0),(img_empty.shape[1],img_empty.shape[0]),(0,255,0),thickness=3) #画线 A->B
cv2.rectangle(img_empty,(0,0),(250,350),(0,0,255),thickness=2,lineType= cv2.FILLED) #画矩形 是否填充
cv2.circle(img_empty,(400,50),30,(255,255,0),thickness=5) #画圆
cv2.putText(img_empty,"OPENCV",(300,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,150,0),thickness=2) #左下角作为起始点坐标写入文本
# cv2.imshow("empty image",img_empty)
# cv2.waitKey(0)

#透视变换
img_poke = cv2.imread("poke.jpg")
print(img_poke.shape)
width,height = 250,350 #生成图片的额图片尺寸
startx,starty = 100,100 #目标区域在生成图片中的左上角起始点
ptsrc = np.float32([[185,70],[234,126],[96,145],[150,205]])  #变换前区域
pttar = np.float32([[startx,starty],[width,starty],[startx,height],[width,height]]) #变换后区域
matrix = cv2.getPerspectiveTransform(ptsrc,pttar) #透视矩阵
imgTransform = cv2.warpPerspective(img_poke,matrix,(width,height)) 
# cv2.imshow("poke image",img_poke)
# cv2.imshow("transfrom image",imgTransform)
# cv2.waitKey(0)

#todo: 可变的水平/竖直堆叠

# 颜色蒙版与滑块
# HSV:0-180,0-255,0-255
maskFlag = False
if maskFlag == True:
    img_car = cv2.imread("car.jpg")
    imgHSV = cv2.cvtColor(img_car,cv2.COLOR_BGR2HSV)

    hmax,hmin,smax,smin,vmax,vmin = 179,0,255,0,255,0
    #滑块移动后的更新函数
    def update_hmax(x):global hmax;hmax = x
    def update_hmin(x):global hmin;hmin = x
    def update_smax(x):global smax;smax = x
    def update_smin(x):global smin;smin = x
    def update_vmax(x):global vmax;vmax = x
    def update_vmin(x):global vmin;vmin = x
    #创建滑块
    #默认最小值是0
    cv2.namedWindow("Slide HSV")
    cv2.createTrackbar("H max","Slide HSV",179,179,update_hmax)
    cv2.createTrackbar("H min","Slide HSV",0,179,update_hmin)
    cv2.createTrackbar("S max","Slide HSV",255,255,update_smax)
    cv2.createTrackbar("S min","Slide HSV",0,255,update_smin)
    cv2.createTrackbar("V max","Slide HSV",255,255,update_vmax)
    cv2.createTrackbar("V min","Slide HSV",0,255,update_vmin)
    while True:
        #生成蒙版，在[lower,upper]范围内的被设置为255（白色），否则设为0
        print(hmin,smin,vmin,hmax,smax,vmax)
        lower = np.array([hmin,smin,vmin])
        upper = np.array([hmax,smax,vmax])
        mask = cv2.inRange(imgHSV,lower,upper)  #返回二维ndarray
        imgResult = cv2.bitwise_and(img_car,img_car,mask = mask) #执行并操作，mask保留白色区域
        cv2.imshow("HSV image",imgHSV)
        cv2.imshow("car image",img_car)
        cv2.imshow("mask image",mask)
        cv2.imshow("result image",imgResult)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

# 轮廓检测与标注
#原图->灰度图->高斯模糊->二值图
img_geometry =cv2.imread("./star.jpg")
imgContour = np.copy(img_geometry)
img_geometryGray = cv2.cvtColor(img_geometry,cv2.COLOR_BGR2GRAY)
img_geometryBlur = cv2.GaussianBlur(img_geometryGray,(7,7),1.0) 
ret ,img_geometryBin  = cv2.threshold(img_geometryBlur,240,255,cv2.THRESH_BINARY) #转化为二值图
# 标注轮廓
def getContours(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) #得到轮廓list
    for cnt in contours:
        area = cv2.contourArea(cnt) #计算所围面积
        if area > 100 : #设置最小阈值消除噪声
            cv2.drawContours(imgContour,cnt,-1,(0,0,255),thickness=2) #绘制轮廓
            cnt_length = cv2.arcLength(cnt,True) #计算周长
            '''
            绘制近似包络线
            '''
            approx =cv2.approxPolyDP(cnt,0.005 * cnt_length,True) #计算近似包裹折线，第二个参数设置贴合程度
            length = len(approx)
            for i in range(len(approx)):
                cv2.line(imgContour, tuple(approx[i][0]), tuple(approx[(i+1)%length][0]), (255,0,0), thickness = 2)
            '''
            绘制凸包
            '''
            hull = cv2.convexHull(cnt)
            length = len(hull)
            for i in range(len(hull)):
                cv2.line(imgContour, tuple(hull[i][0]), tuple(hull[(i+1)%length][0]), (0,255,0), thickness = 2)
            '''
            水平矩形：boundingRect
            最小矩形：minAreaRect
            最小圆形：minEnclosingCircle
            '''
            x,y,w,h = cv2.boundingRect(hull)
            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,0,0),2)
            
imgContour = np.copy(img_geometry) #更新轮廓
getContours(img_geometryBin)
cv2.imshow("geometry image",img_geometry) 
cv2.imshow("blur image",img_geometryBlur) 
cv2.imshow("binary image",img_geometryBin)
cv2.imshow("contour image",imgContour)
cv2.waitKey(0)

#人脸检测
