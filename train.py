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

			# process data in batches
			for batch in Data.BatchIterator(train):

				optimizer.zero_grad()
				
				X, y = data.preprocess(batch)			

				y_hat = model(X)
								
				error = loss(y_hat, y)				
				error.backward()
				optimizer.step()

				print("* loss:", error.data[0])

		# evaluate model
		print("[EVALUATING]")		

	print("\n[DONE]")