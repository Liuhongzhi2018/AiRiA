import cv2 
import os 
import math
# 要提取视频的文件名，隐藏后缀

for name in os.listdir():
    sourceFileName='国网淄博布控终端43号_20191118115839_0233' 
    # 在这里把后缀接上 
    # video_path = os.path.join("", "", sourceFileName+'.MP4')
    video_path = os.path.join("", "", sourceFileName+'.avi')

    times=0 #提取视频的频率，每50帧提取一个 
    outPutDirName='./GW_20191118115839_0233/'
    if not os.path.exists(outPutDirName):     
        # 如果文件目录不存在则创建目录     
        os.makedirs(outPutDirName)

    camera = cv2.VideoCapture(video_path) 
    fps = camera.get(cv2.CAP_PROP_FPS)  
    # FPS:  23.976023976023978

    frameFrequency = 2 * math.ceil(fps)   #输出图片到当前目录vedio文件夹下
    print("FPS: ",fps)
    while True:     
        times += 1     
        res, image = camera.read()     
        if not res:
            print('not res , not image')
            break
        # print("times: ",times, " frameFrequency:",frameFrequency, " times % frameFrequency", times % frameFrequency)
        if times % frameFrequency == 0:
            save_name = outPutDirName + '/' +'GW_20191118115839_0233' +'_' +str(times)+'.jpg'
            cv2.imwrite(save_name , image)         
            print(save_name) 
    print('图片提取结束')
    camera.release()