import numpy as np
import pickle as pkl
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def gaussian_normalize(arr):
    arr = np.array(arr)
    arr = arr - np.mean(arr)
    arr = arr / np.std(arr)
    return np.array(arr)

def load_feature():
	df = pkl.load(open("FeaturesDataFrame.p", "rb"))
	df = df.query('not (bubblecount == -1 & blobpeakfeature > 1)').copy()
	df.reset_index(drop=True, inplace=True)
	return df