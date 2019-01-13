from settings import config
import pandas as pd
from sklearn.model_selection import KFold
from PIL import Image
import torch
from torchvision.transforms import ToTensor

class BatchLoader:
	"""
	Loads batches of a data at a time
	"""

	def __init__(self, data):
		"""
		Inits the batch loader with a dataset to load in batches.
		"""
		self.data = data
		self.batch_size = config["batch_size"]
		self.batch_idx = 0

	def __iter__(self):
		"""
		Required method.
		"""
		return self

	def __next__(self):
		"""
		Serve out fresh batches.
		"""
		upper_idx = (self.batch_idx+1)*self.batch_size
		lower_idx = self.batch_idx*self.batch_size

		if lower_idx < len(self.data):
			self.batch_idx += 1
			return self.preprocess_batch(self.data[lower_idx:upper_idx])

		self.batch_idx = 0
		raise StopIteration

	def preprocess_batch(self, data):
		"""
		Preprocess a batch to go into the model.		
		"""

		# construct X
		X = []
		for path in data["image"]:
			image = Image.open(path)
			image = image.resize((config["image_resize"], config["image_resize"]))
			image = ToTensor()(image)
			image = image.type("torch.FloatTensor")		
			X.append(image)		
		X = torch.stack(X)		

		# construct y
		y = data["malignant"]	
		y = torch.tensor(y.values)
		y = y.type("torch.FloatTensor")
		
		if torch.cuda.is_available():
			X = X.cuda()
			y = y.cuda()

		return X, y

class Data:
	"""
	Loads and iterates over the data.
	"""

	def __init__(self):	
		"""
		Initialize the class
		"""		
		self.data = None
		self.kf = None
		self.fold = None

		self.load_data()
		self.reset_kfold()

	def reset_kfold(self):
		"""
		Resets the k-fold data split
		"""
		self.kf = KFold(n_splits = config["num_k_fold"], shuffle = True, random_state = 2).split(self.data)
		self.fold = 0

	def load_data(self):
		"""
		Loads a csv file into the class.
		"""
		self.data = pd.read_csv(config["dataset"])		
	
	def __iter__(self):
		"""
		Required method.
		"""
		return self	

	def __next__(self):
		"""
		Returns the next k fold iteration.
		"""
		self.fold += 1
		if self.fold > config["num_k_fold"]:
			raise StopIteration

		split = next(self.kf, None)
		train = self.data.iloc[split[0]]
		test =  self.data.iloc[split[1]]
		
		return BatchLoader(train), BatchLoader(test)