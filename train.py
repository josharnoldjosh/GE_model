from settings import config
import data as Data
import model as Model
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':
	
	data = Data.Manager()
	
	av_cosine_sim = 0

	fold_num = 0

	# use k fold cross validation
	for train, test in data:	
		fold_num += 1

		print("\n[STARTING NEW FOLD]")
		model = Model.CNN()
		loss = Model.loss()
		optimizer = Model.optimizer(model)

		error_data = []
		
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

				error_data.append(error.item())

		# evaluate model
		print("\n[EVALUATING]")		
		av_cosine_sim += model.evaluate(test, data)	

		# graph loss		
		plt.plot(error_data)
		plt.ylabel('Model Loss')		
		plt.savefig(str(fold_num)+".png")
		plt.close()

	print("\n[DONE]")
	print("Final averaged cosine similarity over", config["num_k_fold"], "k-folds:", av_cosine_sim/config["num_k_fold"],"\n")