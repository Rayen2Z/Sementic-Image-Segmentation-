import os, glob
import numpy as np
import PIL.Image as pil
import cv2
# without resizing the tensor cannot initiate in our machines because we wont have enough memory to load all the full images


class data(object):
	"""docstring for data"""


	def __init__(self, labels = [0, 1, 2], resize = (256, 256)):
		self.labels = labels
		self.resize = resize
		# Get path for current directory where file exist and data folder should exist next to it
		self.path = os.getcwd() + "/for_sharing"
		# Reading all the instances for both train and test data
		self.Xtrain, self.Xtest = self.x_data()
		# Reading all the targets for instances above
		self.Ytrain, self.Ytest = self.Ytrain_data(), self.Ytest_data()
		# All 4 lists are in order there is no chance for index to mix unless someone makes a mistakes in naming the files


	# Method for reading all instances
	def x_data(self):
		train_data = list()
		test_data = list()
		p = self.path + "/all_segmented"
		# get all the paths for all subfolders, .json files are included but removed using next loop
		folders = glob.glob(p + "/*")
		i = 0
		while i < len(folders):
			if folders[i].endswith(".json"):
				folders.remove(folders[i])
			else:
				i = i + 1
		train_files = list()
		test_files = list()
		# getting all paths for the img.png in each of the folders above test and train separately, since there is a respect 
		# naming manner, only the img index change, sort in here will return the same order as the sort in target methode no matter what
		for temp in folders:
			if temp.rfind("test") != -1:
				test_files.append(glob.glob(temp + "/img.png")[0])
			else:
				train_files.append(glob.glob(temp + "/img.png")[0])
		train_files.sort()
		test_files.sort()
		# Reading the images of shape (675, 1200, 3)
		for f in train_files:
			train_data.append(cv2.resize(np.asarray(pil.open(f)), self.resize))
		for f in test_files:
			test_data.append(cv2.resize(np.asarray(pil.open(f)), self.resize))
		return np.asarray(train_data), np.asarray(test_data)


	# Method for reading train targets
	def Ytrain_data(self):
		# Define main path for train files 
		p = self.path + "/train"
		# Recover paths ofr all subfolders
		folders = glob.glob(p + "/*")
		files = list()
		for temp in folders:
			# Recover paths for all segmented images
			files = files + glob.glob(temp+"/*.png")
		files.sort()
		# Reading the images
		return self.pull(files)

		
	# Same comments as above
	def Ytest_data(self):	
		p = self.path + "/test"
		folders = glob.glob(p + "/*")
		files = list()
		for temp in folders:
			files = files + glob.glob(temp+"/*.png")
		files.sort()
		return self.pull(files)


	def pull(self, images):
		data = list()
		for i in images:
			#read the images, each pixel contain the class it is meant to be
			temp = np.asarray(pil.open(i))
			temp = cv2.resize(temp, self.resize)
			#Generate the one hot vectors
			new_format = np.zeros(temp.shape + (len(self.labels), ))
			for l in self.labels:
				new_format[temp == l, l] = 1
			data.append(new_format)
		return np.asarray(data)


if __name__ == '__main__':
	data = data()
	print(data.Xtrain.shape)
	print(data.Ytrain.shape)
	print(data.Xtest.shape)
	print(data.Ytest.shape)

