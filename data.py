from settings import config
import pandas as pd
from sklearn.model_selection import KFold
import torch
from PIL import Image
from torchvision.transforms import ToTensor

class Manager:
	"""
	Manages data for the project.
	"""

	def __init__(self):				
		self.load_data()
		self.reset()
		return

	def load_data(self):
		# load data frame
		print("[LOADING DATA]")
		self.df = pd.read_csv(config["coords_path"])	
		return

	def reset(self):
		"""Resets the data"""

		# create k fold for cross validation
		self.kf = KFold(n_splits = config["num_k_fold"], shuffle = True, random_state = 2).split(self.df)

		# set the current k fold to 0 
		self.fold = 0
		return

	def __iter__(self):
		return self

	def __next__(self):
		""" iterate through k folds """

		self.fold += 1
		if self.fold > config["num_k_fold"]:
			raise StopIteration

		data = next(self.kf, None)
		train = self.df.iloc[data[0]]
		test =  self.df.iloc[data[1]]

		return train, test

	def preprocess(self, data):
		""" Prepare data to go into the model """

		# construct X
		images = []
		patients = data["Patient"]		
		for file_name in patients:		
			path = config["image_dir"] + file_name + "_raw.tif"	
			image = Image.open(path)			
			image = image.resize((512,512)) # import to resize to same image size
			image = ToTensor()(image)						
			images.append(image)		
		X = torch.stack(images)		
			
		# construct Y
		coords = data.iloc[:,2:]
		y = torch.tensor(coords.values)		

		return X, y

class BatchIterator:
	def __init__(self, data):
		self.data = data
		self.index = 0
		self.batch_size = config["batch_size"]	
		self.batch_idx = 0	
		return

	def __iter__(self):
		return self

	def __next__(self):
		upper_idx = (self.batch_idx+1)*self.batch_size
		lower_idx = self.batch_idx*self.batch_size

		if lower_idx < len(self.data[0]):
			self.batch_idx += 1
			return self.data[0][lower_idx:upper_idx], self.data[1][lower_idx:upper_idx]

		self.batch_number = 0
		raise StopIteration