import numpy as np
import os
import cv2
import datetime

cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    ir_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    assert ir_image.shape == (512, 640), ir_image.shape
    cv2.imshow('ir_image',ir_image)

    now = datetime.datetime.now()
    print(now.hour, now.minute, now.second)
    if now.second == 0:
        fn = 'ir_record/ir_%02d_%02d_%02d.png'%(now.hour, now.minute, now.second)
        if not os.path.exists(fn):
            print(fn)
            cv2.imwrite(fn, ir_image)

    key = cv2.waitKey(100)
    if key == 27:    
        break

cap.release()
cv2.destroyAllWindows()