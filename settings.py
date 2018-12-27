config = {
	"image_dir":"raw/", # path to the folder containing images
	"coords_path":"coords.csv", # path to the coords.csv file
	"batch_size":2, # batch size used during training
	"num_k_fold":5, # number of k folds
	"num_epoch":10, # number of epochs
	"num_input_channels":20, # number of tiff files in a stack to load
	"learning_rate":0.001,
	"cuda":True
}