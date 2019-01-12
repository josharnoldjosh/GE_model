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

	def load_image(self, path):
		images = Image.open(path)
		result = []
		n = config["num_input_channels"] # must standarize input		
		for i in range(n):		
			images.seek(i)
			image = images.resize((512, 512))						
			image = ToTensor()(image)
			image = image.type("torch.FloatTensor")
			result.append(image)
		output = torch.cat(result)
		#print(output.size())
		return output


	def preprocess(self, data):
		""" Prepare data to go into the model """

		# construct X
		X = []
		patients = data["Patient"]		
		for file_name in patients:		
			path = config["image_dir"] + file_name + "_raw.tif"	
			image = self.load_image(path)								
			X.append(image)	
		X = torch.stack(X)
		#print(X.size())		

		# construct Y
		coords = data.iloc[:,2:]
		y = torch.tensor(coords.values)
		y = y.type("torch.FloatTensor")
		
		if torch.cuda.is_available() and config["cuda"]:
			X = X.cuda()
			y = y.cuda()

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

		if lower_idx < len(self.data):
			self.batch_idx += 1
			return self.data[lower_idx:upper_idx]

		self.batch_number = 0
		raise StopIteration