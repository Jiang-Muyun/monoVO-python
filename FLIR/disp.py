import numpy as np
import os
import cv2
import glob

regx = 'ir_record/*.png'
path_list = list(glob.glob(regx))
path_list.sort()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 60.0, (640, 512))

for path in path_list:
    ir_image = cv2.imread(path, 1)
    assert ir_image.shape[:2] == (512, 640), ir_image.shape

    fn = path.split('\\')[-1]
    hour,minute = fn[3:5], fn[6:8]
    text = 'time: %s:%s'%(hour, minute)
    cv2.putText(ir_image, text,(200,60),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),4)
    cv2.imshow('ir_image',ir_image)
    
    out.write(ir_image)

    key = cv2.waitKey(10)
    if key == 27:
        break

out.release()