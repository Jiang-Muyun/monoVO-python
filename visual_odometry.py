import numpy as np 
import cv2

INIT_STATE_1 = 0
INIT_STATE_2 = 1
VO_NORMAL = 2
kMinNumFeature = 1500

lk_params = dict(winSize  = (21, 21), 
				#maxLevel = 3,
             	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def featureTracking(image_ref, image_cur, px_ref):
	# shape: [k,2] [k,1] [k,1]
	kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  
	st = st.reshape(-1)
	kp1 = px_ref[st == 1]
	kp2 = kp2[st == 1]
	print(kp1.shape, kp2.shape, st.shape, '---', image_ref.shape, image_cur.shape)
	return kp1, kp2


class PinholeCamera:
	def __init__(self, width, height, fx, fy, cx, cy,  k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
		self.width = width
		self.height = height
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy
		self.distortion = (abs(k1) > 0.0000001)
		self.d = [k1, k2, p1, p2, k3]


class VisualOdometry:
	def __init__(self, cam, annotations):
		self.state = 0
		self.cam = cam
		self.new_frame = None
		self.last_frame = None
		self.cur_R = None
		self.cur_t = None
		self.px_ref = None
		self.px_cur = None
		self.focal = cam.fx
		self.pp = (cam.cx, cam.cy)
		self.trueX, self.trueY, self.trueZ = 0, 0, 0
		self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
		with open(annotations) as f:
			self.annotations = f.readlines()

	def getAbsoluteScale(self, frame_id):  #specialized for KITTI odometry dataset
		ss = self.annotations[frame_id-1].strip().split()
		x_prev = float(ss[3])
		y_prev = float(ss[7])
		z_prev = float(ss[11])
		ss = self.annotations[frame_id].strip().split()
		x = float(ss[3])
		y = float(ss[7])
		z = float(ss[11])
		self.trueX, self.trueY, self.trueZ = x, y, z
		return np.sqrt((x - x_prev)**2 + (y - y_prev)**2 + (z - z_prev)**2)

	def processFrame(self, frame_id):
		self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
		E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		_, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
		
		absolute_scale = self.getAbsoluteScale(frame_id)
		if absolute_scale > 0.1:
			self.cur_t = self.cur_t + absolute_scale*self.cur_R.dot(t) 
			self.cur_R = R.dot(self.cur_R)

		if self.px_ref.shape[0] < kMinNumFeature:
			print('new keyframe', frame_id, self.px_ref.shape)
			self.px_cur = self.detector.detect(self.new_frame)
			self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)

		self.px_ref = self.px_cur

	def update(self, img, frame_id):
		assert(img.ndim==2 and img.shape[0]==self.cam.height and img.shape[1]==self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
		self.new_frame = img

		if self.state == INIT_STATE_1:
			self.px_ref = self.detector.detect(self.new_frame)
			self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
			self.state = INIT_STATE_2

		elif self.state == INIT_STATE_2:
			self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
			E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
			_, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
			self.px_ref = self.px_cur
			self.state = VO_NORMAL

		elif self.state == VO_NORMAL:
			self.processFrame(frame_id)

		self.last_frame = self.new_frame
