from settings import config
import data as Data
import model as Model

if __name__ == '__main__':
	
	data = Data.Manager()

	model = Model.CNN()

	loss = Model.loss()

	optimizer = Model.optimizer(model)

	# use k fold cross validation
	for train, test in data:	

		print("\n[STARTING NEW FOLD]")	
		
		# train model
		for epoch in range(config["num_epoch"]):
			
			print("[EPOCH]", epoch+1)
			
			for batch in Data.BatchIterator(train):

				optimizer.zero_grad()
				
				X, y = data.preprocess(batch)			

				y_hat = model(X)
								
				error = loss(y_hat, y)				
				error.backward()
				optimizer.step()

				print("	* loss:", error.item())

		# evaluate model
		print("[EVALUATING]")		

		for batch in Data.BatchIterator(test):
			X, y = data.preprocess(batch)
			y_hat = model(X)
			result = torch.pairwise_distance(y_hat, y)
			print("	* result:", result)


	print("\n[DONE]")