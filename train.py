from settings import config
import data as Data

if __name__ == '__main__':

	# load data manager
	data = Data.Manager()

	# use k fold cross validation
	for train, test in data:	

		print("\n[STARTING NEW FOLD]")	
		
		# train model
		for epoch in range(config["num_epoch"]):
			
			print("[EPOCH]", epoch+1)

			X, y = data.preprocess(train)			

			for batch_X, batch_y in Data.BatchIterator((X, y)):
				


		# evaluate model
		print("[EVALUATING]")		

	print("\n[DONE]")