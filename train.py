from settings import config
import data as Data
from model import Model

if __name__ == '__main__':
	
	data = Data.Manager()

	model = Model()

	# use k fold cross validation
	for train, test in data:	

		print("\n[STARTING NEW FOLD]")	
		
		# train model
		for epoch in range(config["num_epoch"]):
			
			print("[EPOCH]", epoch+1)

			X, y = data.preprocess(train)			

			# process data in batches
			for batch_X, batch_y in Data.BatchIterator((X, y)):
				model(X)
				print("hi")


		# evaluate model
		print("[EVALUATING]")		

	print("\n[DONE]")