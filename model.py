from settings import config
import torch
import torch.nn as nn
import torch.nn.functional as F 

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		self.conv1 = nn.Conv2d(config["num_input_channels"], 40, kernel_size=(5, 5), stride=1, padding=(1,1))		
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		
		if torch.cuda.is_available() and config["cuda"]:
			self.conv1.cuda()		
			self.pool1.cuda()	

		return

	def forward(self, x):
		
		x = self.conv1(x)	

		x = F.relu(x)

		x = self.pool1(x)

		print(x.size())

		return x

		