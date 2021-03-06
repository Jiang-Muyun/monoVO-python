import numpy as np 
import cv2

from visual_odometry import PinholeCamera, VisualOdometry

cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
vo = VisualOdometry(cam, 'E:/Dataset/KITTI/odometry/dataset/poses/00.txt')

traj = np.zeros((600,600,3), dtype=np.uint8)

for img_id in range(4541):
	img = cv2.imread('E:/Dataset/KITTI/odometry/dataset/sequences/00/image_2/%06d.png'%(img_id), 0)

	vo.update(img, img_id)
	if vo.state == 2:
		x, y, z = vo.get_T()
		draw_x, draw_y = int(x)+290, int(z)+90
		true_x, true_y = int(vo.trueX)+290, int(vo.trueZ)+90

		cv2.circle(traj, (draw_x,draw_y), 1, (img_id*255/4540,255-img_id*255/4540,0), 1)
		cv2.circle(traj, (true_x,true_y), 1, (0,0,255), 2)
		cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
		text = "Coordinates: x=%.2fm y=%.2fm z=%.2fm"%(x,y,z)
		cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

		cv2.imshow('Road facing camera', img)
		cv2.imshow('Trajectory', traj)
	
	if 27 == cv2.waitKey(1):
		break

