from settings import config
from data import Data
from model import Model
from model import Loss
from model import Optimizer
import torch

data = Data()

def evaluate(model, test):
	i = 0
	score = 0
	for X, y in test:
		i += 1
		y_hat = model(X)
		if torch.cuda.is_available():
			y_hat = y_hat.detach().cpu().numpy()
			y = y.detach().cpu().numpy()
		else:
			y_hat = y_hat.detach().numpy()
			y = y.detach().numpy()		
		y_hat = [1 if x > 0.5 else 0 for x in y_hat]
		y = [1 if x > 0.5 else 0 for x in y]
		result = (sum([1 if tup[0] == tup[1] else 0 for tup in zip(y_hat, y)])/len(y))*100
		score += result
	score = score/i

	result = "\n\n[Accuracy: %.2f%%]" % (score)
	print(result)
	return score

def train():
	scores = []
	for train, test in data:
		
		model = Model()
		loss = Loss()
		optimizer = Optimizer(model)
		batch_num = 0

		print("\n\nStarting new K-fold")

		for epoch in range(1, config["num_epoch"]):

			print("\n\nStarting epoch", epoch)

			for X, y in train:
				
				optimizer.zero_grad()
					
				y_hat = model(X)

				error = loss(y_hat, y)
				
				if batch_num == 0 or batch_num % config["display_rate"] == 0:
					if torch.cuda.is_available():
						cost = "Cost: %.4f" % (error.detach().cpu().numpy())
						print(cost)
					else:
						cost = "Cost: %.4f" % (error.detach().numpy())
						print(cost)
				batch_num += 1

				error.backward()

				optimizer.step()
				
			evaluate(model, test)
		scores.append(evaluate(model, test))
	return scores

if __name__ == '__main__':
	scores = train()
	result = sum(scores)/len(scores)	
	print("\n\nFinal averaged accuracy: %.2f%%" % result)