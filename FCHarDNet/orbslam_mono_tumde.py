
#import os.path
import orbslam2
import time
import torch
import argparse
import os
import cv2
import sys
import numpy as np

from collections import OrderedDict
from models import get_model
from skimage import img_as_ubyte


#cv2.namedWindow('seg', cv2.WINDOW_NORMAL) 
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
def convert_state_dict(state_dict):
	"""Converts a state dict saved from a dataParallel module to normal
	   module state_dict inplace
	   :param state_dict is the loaded DataParallel model_state
	"""
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		name = k[7:]  # remove `module.`
		new_state_dict[name] = v
	return new_state_dict
def decode_segmap(temp,label_colours):
	r = temp.copy()
	g = temp.copy()
	b = temp.copy()
	for l in range(0, 19):
		r[temp == l] = label_colours[l][0]
		g[temp == l] = label_colours[l][1]
		b[temp == l] = label_colours[l][2]
#		print(r,',',g,',',b)
	print('hi')
	rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
	rgb[:, :, 0] = r / 255.0
	rgb[:, :, 1] = g / 255.0
	rgb[:, :, 2] = b / 255.0
	return rgb

def process_img(img, size, device, model, msktp):
#	print("Read Input Image from : {}".format(img_path))

#	img = cv2.imread(img_path)
	img_resized = cv2.resize(img, (size[0], size[1]))  # uint8 with RGB mode
	img = img_resized.astype(np.float16)

	# norm
	value_scale = 255
	mean = [0.406, 0.456, 0.485]
	mean = [item * value_scale for item in mean]
	std = [0.225, 0.224, 0.229]
	std = [item * value_scale for item in std]
	img = (img - mean) / std

	# NHWC -> NCHW
	img = img.transpose(2, 0, 1)
	img = np.expand_dims(img, 0)
	img = torch.from_numpy(img).float()

	images = img.to(device)
	outputs = model(images)
	colors = [  # [  0,   0,   0],
		[128, 64, 128],
		[244, 35, 232],
		[70, 70, 70],
		[102, 102, 156],
		[190, 153, 153],
		[153, 153, 153],
		[250, 170, 30],
		[220, 220, 0],
		[107, 142, 35],
		[152, 251, 152],
		[0, 130, 180],
		[220, 20, 60],
		[255, 0, 0],
		[0, 0, 142],
		[0, 0, 70],
		[0, 60, 100],
		[0, 80, 100],
		[0, 0, 230],
		[119, 11, 32],
	]
	
	colors2 = [
		[255, 255, 255],
		[255, 255, 255],
		[255, 255, 255],
		[255, 255, 255],
		[255, 255, 255],
		[255, 255, 255],
		[255, 255, 255],
		[255, 255, 255],
		[255, 255, 255],
		[255, 255, 255],
		[255, 255, 255],
		[255, 255, 255],
		[255, 255, 255],
		[255, 255, 255],
		[255, 255, 255],
		[255, 255, 255],
		[255, 255, 255],
		[255, 255, 255],
		[255, 255, 255],
	]
	
	label_colours = dict(zip(range(19), colors))
	label_colours2 = dict(zip(range(19), colors))
	label_colours2[int(msktp)] = [0, 0, 0]
#	print (label_colours)
#	print('Output shape: ',outputs.shape)
	pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)

	decoded = decode_segmap(temp=pred,label_colours=label_colours2)

	return img_resized, decoded


def init_model(model_path):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	n_classes = 19
	# Setup Model
	model = get_model({"arch": "hardnet"}, n_classes)
	state = convert_state_dict(torch.load(model_path, map_location=device)["model_state"])
	model.load_state_dict(state)
	model.eval()
	model.to(device)
	return device, model
	

def seg_process(img_in, masktp):
	#蒙版生成
	x,y = process_img(img_in, [640,360], device, model.cuda(), masktp)	
	
	#转换蒙版值类型，平滑化，单通道化
	z = img_as_ubyte(y)													
	z = cv2.blur(cv2.resize(z, (img_in.shape[1], img_in.shape[0])), (5, 5))
	gz = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY) 
	
	#形态核定义，膨胀（黑色腐蚀）
	# ~ kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	# ~ msk = cv2.dilate(gz, kernel, iterations = 5)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	# ~ msk = cv2.dilate(gz, kernel, iterations = 5)
	msk = cv2.morphologyEx(gz, cv2.MORPH_CLOSE, kernel, iterations = 5) 
	#套蒙版	
	image2 = cv2.bitwise_and(img_in, img_in, mask=msk)
#	print('sttt = ', int(sttt))

	cv2.imshow('img2', image2)
	cv2.waitKey(1)

#	
#	print("hi2")
#	k = cv2.waitKey(1)
#	if k == 27:
#		exit(0)
	return image2

device,model = init_model("pretrained/hardnet70_cityscapes_model.pkl")

def main(vocab_path, settings_path, sequence_path, masktp):

	rgb_filenames, timestamps = load_images(sequence_path)
	num_images = len(timestamps)

	slam = orbslam2.System(vocab_path, settings_path, orbslam2.Sensor.MONOCULAR)
	
#	slam.set_use_viewer(True)
	slam.set_use_viewer(False)
	slam.initialize()

	times_track = [0 for _ in range(num_images)]
	print('-----')
	print('Start processing sequence ...')
	print('Images in the sequence: {0}'.format(num_images))

	for idx in range(num_images):
		image = cv2.imread(os.path.join(sequence_path, rgb_filenames[idx]), cv2.IMREAD_UNCHANGED)
		image2 = seg_process(image, masktp)
		'''		
		x,y = process_img(image,[640,360],device,model.cuda(), masktp)
		z = img_as_ubyte(y)
		z = cv2.resize(z, (image.shape[1],image.shape[0]))
		gz = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY) 
		image2 = cv2.bitwise_and(image, image, mask=gz)
		cv2.imshow('img2',z)
		print("hi2")
		k = cv2.waitKey(1)
		if k == 27:
			exit(0)
		'''
		tframe = timestamps[idx]

		if image is None:
			print("failed to load image at {0}".format(os.path.join(sequence_path, rgb_filenames[idx])))
			return 1

		t1 = time.time()
		slam.process_image_mono(image, tframe)
		t2 = time.time()

		ttrack = t2 - t1
		times_track[idx] = ttrack

		t = 0
		if idx < num_images - 1:
			t = timestamps[idx + 1] - tframe
		elif idx > 0:
			t = tframe - timestamps[idx - 1]

		if ttrack < t:
			time.sleep((t - ttrack)*0.0004)
 
	save_trajectory(slam.get_trajectory_points(), 'trajectory.txt')

	slam.shutdown()

	times_track = sorted(times_track)
	total_time = sum(times_track)
	print('-----')
	print('median tracking time: {0}'.format(times_track[num_images // 2]))
	print('mean tracking time: {0}'.format(total_time / num_images))

	return 0


def load_images(path_to_association):
	rgb_filenames = []
	timestamps = []
	with open(os.path.join(path_to_association, 'rgb.txt')) as times_file:
		for line in times_file:
			if len(line) > 0 and not line.startswith('#'):
				t, rgb = line.rstrip().split(' ')[0:2]
				rgb_filenames.append(rgb)
				timestamps.append(float(t))
	return rgb_filenames, timestamps


def save_trajectory(trajectory, filename):
	with open(filename, 'w') as traj_file:
		traj_file.writelines('{time} {r00} {r01} {r02} {t0} {r10} {r11} {r12} {t1} {r20} {r21} {r22} {t2}\n'.format(
			time=repr(stamp),
			r00=repr(r00),
			r01=repr(r01),
			r02=repr(r02),
			t0=repr(t0),
			r10=repr(r10),
			r11=repr(r11),
			r12=repr(r12),
			t1=repr(t1),
			r20=repr(r20),
			r21=repr(r21),
			r22=repr(r22),
			t2=repr(t2)
		) for stamp, r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2 in trajectory)





if __name__ == '__main__':
	if len(sys.argv) != 5:
		print('Usage: ./orbslam_mono_tum path_to_vocabulary path_to_settings path_to_sequence mask_object_type viewer_setting')
	main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
