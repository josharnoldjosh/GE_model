config = {
	# input parameters
	'dataset':'dataset.csv', # the filename of the dataset CSV

	# training parameters
	'learning_rate':0.00001, # learning rate of the model
	'num_epoch':20, # number of epoches
	'image_resize':256, # the width & height of resize imageds into the model
	'num_k_fold':5, # the number of k folds for CV
	'batch_size':20, # size of batches

	# etc
	'display_rate':20 # display the loss at every x batch
}