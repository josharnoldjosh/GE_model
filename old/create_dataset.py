from j.osh import *
import pandas as pd
from PIL import Image
from random import shuffle
from random import randint
import numpy
import uuid

def load_csv():
	"""
	Loads the CSV file for reading coordinates.
	"""
	return pd.read_csv('coords.csv')

def load_img(filename):
	"""
	Loads an Image with PIL for a given filename.
	"""
	img = Image.open(filename)
	return img

def crop_img(img, x1, x2, y1, y2, z):
	"""
	Crops an image with given coordinates.
	"""
	img.seek(z)	
	img = img.crop((x1, y1, x2, y2))	
	return img

def n_frames(img):
	"""
	Returns the number of frames in an image stack.
	"""
	img.seek(0)
	i = 1
	while img:
		try:
			img.seek(i)
		except:
			break
		i += 1
	return i

def save_image(image):
	"""
	Will save an image to file, then return its filename.
	"""

	# generate unique token
	unq = 'cropped/' + str(uuid.uuid4().hex[:6].upper()) + '.jpg'

	# ensure cropped exists
	ensure_dir("cropped/") 

	image.mode = 'I'
	image.point(lambda i:i*(1./256)).convert('L').save(unq)

	return unq

def add_crop_to_row(cropped_img, malignant, x1, x2, y1, y2, z):
	"""
	Takes an image and its values and formats it into a row in an array.
	"""
	unq = save_image(cropped_img)

	return [{'image':unq, 'malignant':malignant,
	'x1':x1, 'x2':x2, 'y1':y1, 'y2':y2, 'z':z,
	'image_size':cropped_img.size}]

def crop_malignant(img, x1, x2, y1, y2, z1, z2):
	"""
	Will crop out all the malignant images and return them in an array.
	"""
	output = []

	# loop through "z" slices
	z_range = range(int(z1), int(z2)+1)
	for z in z_range:			

		# crop image
		cropped_img = crop_img(img, x1, x2, y1, y2, z)

		# add to dataframe
		output += add_crop_to_row(cropped_img, 1, x1, x2, y1, y2, z)
	return output

def random_z(img):
	"""
	Return a random z coordinate
	"""
	return randint(0, n_frames(img)-1)

def gen_crop_coords(img, x1, x2, y1, y2, z, minx=50, miny=50, maxx=200, maxy=200):
	"""
	Will offset a given images x and y coordinates to give a random crop.
	"""	

	# get original crop size
	orig_crop_size = (abs(x1-x2), abs(y1-y2))

	# calculate a new crop size that is either big or small depending on size of orig crop
	div_const = 5
	new_crop_size = ((img.size[0] - orig_crop_size[0])/div_const, (img.size[1] - orig_crop_size[1])/div_const)

	# initialize to zero
	X1, X2, Y1, Y2 = 0, 0, 0, 0

	padding = 8

	x_bound = img.size[0]/2
	if x1 > x_bound and x2 > x_bound:
		# generate to left
		X1 = randint(padding, int(img.size[0]-new_crop_size[0]-max(x1, x2))-padding)
		X2 = X1 + new_crop_size[0]			
	else:
		# generate to right		
		X1 = randint(int(max(x1, x2))+padding, int(img.size[0]-new_crop_size[0])-padding)
		X2 = X1 + new_crop_size[0]

	y_bound = img.size[1]/2
	if y1 > y_bound and y2 > y_bound:
		# generate below
		Y1 = randint(padding, int(img.size[1]-new_crop_size[1]-max(y1, y2))-padding)
		Y2 = Y1 + new_crop_size[1]	
	else:
		# generate above
		Y1 = randint(int(max(y1, y2))+padding, int(img.size[1]-new_crop_size[1])-padding)
		Y2 = Y1 + new_crop_size[1]


	return X1, X2, Y1, Y2

def crop_benign(img, x1, x2, y1, y2, z1, z2):
	"""
	Will crop out random benign images - need to edit this!!!!
	"""	
	output = []

	# we want to crop three images
	for i in range(0, 3):

		# generate random z 
		z = random_z(img)

		# get coordinates
		X1, X2, Y1, Y2 = gen_crop_coords(img, x1, x2, y1, y2, z)

		# crop image
		cropped_img = crop_img(img, X1, X2, Y1, Y2, z)			

		# add to dataframe
		output += add_crop_to_row(cropped_img, 0, X1, X2, Y1, Y2, z)

	return output

if __name__ == '__main__':

	# load data frame
	df = load_csv()

	# output
	output = []

	# Loop over rows in data frame
	for index, row in df.iterrows():

		# extract data
		img_name, x1, x2, y1, y2, z1, z2 = row["Patient"], row["x1"], row["x2"], row["y1"], row["y2"], row["z1"], row["z2"]		 
		img = load_img("raw/"+img_name+"_raw.tif")		

		# crop malignant
		output += crop_malignant(img, x1, x2, y1, y2, z1, z2)

		# crop benign
		output += crop_benign(img, x1, x2, y1, y2, z1, z2)
	
	# create dataframe
	result = pd.DataFrame(output)

	# save dataframe
	result.to_csv("dataset.csv")

	print("script done.")