import cv2
import os
import math

sourceFileName='YI015601_crop' 
video_path = os.path.join("", "", sourceFileName+'.mp4') 

times=0
outPutDirName='./YI015601_crop/'
if not os.path.exists(outPutDirName):     
    os.makedirs(outPutDirName)

camera = cv2.VideoCapture(video_path) 
fps = camera.get(cv2.CAP_PROP_FPS)  
# FPS:  23.976023976023978

frameFrequency = math.ceil(1/fps)  # 2 * math.ceil(fps) 
print("FPS: ",fps)
while True:     
    times += 1     
    res, image = camera.read()     
    if not res:
        print('not res , not image')
        break
        # print("times: ",times, " frameFrequency:",frameFrequency, " times % frameFrequency", times % frameFrequency)
    if times % frameFrequency == 0:
        save_name = outPutDirName + '/' +sourceFileName +'_' +str(times).zfill(6)+'.jpg'
        cv2.imwrite(save_name , image)         
        print(save_name) 
print('Finish')
camera.release()