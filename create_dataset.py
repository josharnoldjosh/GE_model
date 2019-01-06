from j.osh import *
import pandas as pd
from PIL import Image
from random import shuffle
from random import randint

def load_csv():
	return pd.read_csv('coords.csv')

def load_img(filename):
	img = Image.open(filename)
	return img

def crop_img(img, x1, x2, y1, y2, z):
	img.seek(z)
	return img.crop((x1, x2, y1, y2))

def n_frames(img):
	img.seek(0)
	i = 1
	while img:
		try:
			img.seek(i)
		except:
			break
		i += 1
	return i

def get_possible_z(img, z_range):
	possible_z = [x for x in list(range(0, n_frames(img))) if x not in z_range]
	shuffle(possible_z)
	return possible_z

def get_offset(img, x1, x2, y1, y2):
	x_range = x2 - x1
	y_range = y2 - y1
	return randint(0, img.size[0]-x_range), x_range, randint(0, img.size[1]-y_range), y_range

if __name__ == '__main__':
	df = load_csv()
	output = []

	# Loop over rows in df
	for index, row in df.iterrows():

		# extract data
		img_name, x1, x2, y1, y2, z1, z2 = row["Patient"], row["x1"], row["x2"], row["y1"], row["y2"], row["z1"], row["z2"]		 
		img = load_img("raw/"+img_name+"_raw.tif")		

		# loop through "z" slices
		z_range = range(int(z1)-1, int(z2))
		for z in z_range:			

			# crop image
			cropped_img = crop_img(img, x1, x2, y1, y2, z)

			# add to dataframe
			output.append({'image':cropped_img, 'cancer':1, 'x1':x1, 'x2':x2, 'y1':y1, 'y2':y2, 'z1':z1, 'z2':z2})

		# crop extra "x" amount of images "without" cancer
		i = 0
		for z in get_possible_z(img, z_range):			

			if i > 2: # the max number of extra "non-cancer" crops
				break

			# get a random x & y coordinate thats the same size
			x1, x2, y1, y2 = get_offset(img, x1, x2, y1, y2)
						
			# crop image
			cropped_img = crop_img(img, x1, x2, y1, y2, z)			

			# add to dataframe
			output.append({'image':cropped_img, 'cancer':0, 'x1':x1, 'x2':x2, 'y1':y1, 'y2':y2, 'z1':z1, 'z2':z2})

			# increment 1
			i += 1
	
	result = pd.DataFrame(output)
	result.to_csv("dataset.csv", sep='\t') # save this in different format
	print("script done.")


