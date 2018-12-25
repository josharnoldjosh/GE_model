from settings import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def loss():
	return torch.nn.MSELoss()

def optimizer(model):	
	return optim.Adam(model.parameters(), lr=config["learning_rate"])

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()

		self.conv1 = nn.Conv2d(config["num_input_channels"], 40, kernel_size=(5, 5), stride=1, padding=(1,1))		
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		self.fc1 = torch.nn.Linear(2601000, 64)
		self.fc2 = torch.nn.Linear(64, 6)
		
		if torch.cuda.is_available() and config["cuda"]:
			self.conv1.cuda()		
			self.pool1.cuda()	
			self.fc1.cuda()
			self.fc2.cuda()

		return

	def forward(self, x):
		
		x = self.conv1(x)	

		x = F.relu(x)

		x = self.pool1(x)

		x = x.view(-1, 2601000) # resize to one dimension

		x = F.relu(self.fc1(x))
        
		x = self.fc2(x)
		
		return x

		