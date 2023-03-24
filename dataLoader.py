import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
import random
import tifffile
import cv2
from base_dataloader import get_transform
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.transforms as transform

width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351
width_to_focal[1226] = 711.3722
baseline = 0.54

class KITTIDataloader(Dataset):

	__left = []
	__right = []
	__disp = []
	__rdisp = []
	__conf = []
	__lidar = []

	def __init__(self):
		self.filename = "./utils/KITTI_train_file"
		self.img_root = "/KITTI/Train/Left/"
		self.disp_root = "/KITTI/Train/Disp/"
		self.toPIL = transform.ToPILImage()
		self.toTensor = transform.ToTensor()

		arr = open(self.filename, "rt").read().split("\n")[:-1]

		n_line = open(self.filename).read().count('\n')

		for line in range(n_line):
			self.__left.append(self.img_root + arr[line] + '.png')
			self.__disp.append(self.disp_root + arr[line] + '.png')

	def __getitem__(self, index):
		img1 = Image.open(self.__left[index]).convert('RGB')
		img2 = Image.open(self.__disp[index])

		transform = get_transform()
		img1 = transform(img1)

		img2_ori = self.load_image2(img2)
		img2_disp_ori = self.load_image2_disp(img2)

		img2_ori = self.toTensor(img2_ori)
		img2_disp_ori = self.toTensor(img2_disp_ori)


		input_dict = {'left_img':img1, 'disp_img':img2_ori, 'disparity_img' : img2_disp_ori}

		return input_dict


	def load_image2(self,image):

		w, h = image.size
		imx_t = np.asarray(image)
		imx_t = imx_t / 256 
		imx_t = baseline * width_to_focal[w] / imx_t
		nmimg = Image.fromarray(imx_t)
		nmimg = nmimg.resize((480, 192))

		return nmimg

	def load_image2_disp(self,image):

		imx_t = np.asarray(image)
		imx_t = imx_t / 256
		nmimg = Image.fromarray(imx_t)
		nmimg = nmimg.resize((480, 192))

		return nmimg

class KITTIValDataloader(Dataset):

	__left = []
	__disp = []

	def __init__(self, filename, img_root, disp_root):
		self.filename = filename
		self.img_root = img_root
		self.disp_root = disp_root

		arr = open(self.filename, "rt").read().split("\n")[:-1]

		n_line = open(self.filename).read().count('\n')

		for line in range(n_line):
			self.__left.append(self.img_root + arr[line])
			self.__disp.append(self.disp_root + arr[line])


	def __getitem__(self, index):
		img1 = Image.open(self.__left[index]).convert('RGB')
		img2 = Image.open(self.__disp[index])

		transform = get_transform()
		img1 = transform(img1)
		img2 = self.load_image2(img2)
		img2 = torch.FloatTensor(img2)

		input_dict = {'left_img':img1, 'disp_img':img2}

		return input_dict

	def load_image2(self,image):

		w, h = image.size
		imx_t = (np.asarray(image)) / 256
		imx_t = baseline * width_to_focal[w] / imx_t
		nmimg = Image.fromarray(imx_t)
		nmimg = nmimg.resize((480, 192))
		imx_t = (np.asarray(nmimg))

		return imx_t

	def augument_image_pair(self, left_image, disp_image):

		left_image = np.array(left_image)
		random_gamma = random.uniform(0.8, 0.9)
		left_image_aug  = left_image  ** random_gamma

		random_brightness = random.uniform(0.5, 0.8)
		left_image_aug  =  left_image_aug * random_brightness

		left_image_aug = Image.fromarray(np.uint8(left_image_aug))


		return left_image_aug, disp_image

	def __len__(self):
		return len(self.__left)



