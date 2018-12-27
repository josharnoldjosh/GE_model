from settings import config
import data as Data
import model as Model
import torch

if __name__ == '__main__':
	
	data = Data.Manager()

	# use k fold cross validation
	av_cosine_sim = 0
	for train, test in data:	

		print("\n[STARTING NEW FOLD]")
		model = Model.CNN()
		loss = Model.loss()
		optimizer = Model.optimizer(model)
		
		# train model
		for epoch in range(config["num_epoch"]):
			
			print("\n[EPOCH]", epoch+1)
			
			i = 0

			for batch in Data.BatchIterator(train):

				i += 1

				optimizer.zero_grad()
				
				X, y = data.preprocess(batch)			

				y_hat = model(X)
								
				error = loss(y_hat, y)		

				error.backward()
				
				optimizer.step()

				if i%8 == 0:
					print("	* loss:", error.item())

		# evaluate model
		print("\n[EVALUATING]")		
		av_cosine_sim += model.evaluate(test, data)			

	print("\n[DONE]")
	print("Final averaged cosine similarity over", config["num_k_fold"], "k-folds:", av_cosine_sim/config["num_k_fold"],"\n")