import time
import cv2
import numpy as np

cap = cv2.VideoCapture('data/FLIR_Car_Test.mp4')

class Thermal_FFC():
    def __init__(self):
        self.FFC_P1 = (620, 50)
        self.FFC_P2 = (629, 59)
        self.FFC_Center = (625, 55)
        self.FFC_State = False
        self.FFC_State_last = False

        self.freeze_counter = 0
        self.freeze_frame = None
        self.last_frame = None

    def process_frame(self, frame):
        test_pt = frame[self.FFC_Center[1], self.FFC_Center[0]].astype(np.int32)
        self.FFC_State = (test_pt[1] - test_pt[0]) > 10

        if self.FFC_State == False and self.FFC_State_last==True:
            self.freeze_counter = 3
            self.freeze_frame = self.last_frame
        self.FFC_State_last = self.FFC_State

        frame = frame[:, :, 1].copy()
        # frame[50:60, 620:630] = frame[40:50, 620:630]
        frame[50:55, 620:630] = frame[45:50, 620:630]
        frame[55:60, 620:630] = frame[60:65, 620:630]

        if self.freeze_counter > 0:
            self.freeze_counter -= 1
            return self.freeze_frame
        else:
            self.last_frame = frame
            return frame

thermal_handle = Thermal_FFC()
index = 0
while(True):
    ret, frame = cap.read()
    index += 1
    if not ret:
        break
    # if index < 15:
    #     continue

    start = time.time()
    cv2.imshow('frame', frame)
    frame = thermal_handle.process_frame(frame)
    cv2.imshow('process_frame', frame)

    if cv2.waitKey(100) == 27:
        break
