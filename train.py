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

			# process data in batches
			for batch in Data.BatchIterator(train):		
				
				X, y = data.preprocess(batch)			

				model(X)

				print("*data passed")


		# evaluate model
		print("[EVALUATING]")		

	print("\n[DONE]")