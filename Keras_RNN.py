import numpy as np

WEIGHTS_PATH = '/Users/WenHsu/Documents/ECE 239 NN/Project/project/saved_models/Keras_RNN.h5'
ARCHITECTURE_PATH = '/Users/WenHsu/Documents/ECE 239 NN/Project/project/saved_models/Keras_RNN_Architecture.json'

class RNN(object):
	def __init__(self, print_data = True):
		self.model = None
		self.X_train_valid = None
		self.y_train_valid = None
		self.X_test = None
		self.y_test = None
		self.person_test = None
		self.person_train_valid = None
		self.num_classes = None

		self.load_data(print_data)
		self.history = None

	def load_data(self, print_data = True):
		self.X_test = np.load("X_test.npy")
		self.y_test = np.load("y_test.npy")
		self.person_train_valid = np.load("person_train_valid.npy")
		self.X_train_valid = np.load("X_train_valid.npy")
		self.y_train_valid = np.load("y_train_valid.npy")
		self.person_test = np.load("person_test.npy")

		# Not use the last 3 of the 25 electrodes, which are EOG (rather than EEG) electrodes
		self.X_train_valid = self.X_train_valid[:,:-3,:]
		self.X_test = self.X_test[:,:-3,:]

		# Turn the data into 0 mean and 1 var
		from sklearn import preprocessing
		for i in range(self.X_train_valid.shape[0]):
			self.X_train_valid[i] = preprocessing.scale(self.X_train_valid[i])
		for i in range(self.X_test.shape[0]):
			self.X_test[i] = preprocessing.scale(self.X_test[i])
		
		# Change the timestep to the second column
		self.X_train_valid = np.transpose(self.X_train_valid, (0, 2, 1))
		self.X_test = np.transpose(self.X_test, (0, 2, 1))

		# Modify the y to categorical form
		from keras.utils import np_utils
		self.num_classes = np.unique(self.y_train_valid).size
		self.y_train_valid = self.y_train_valid - min(self.y_train_valid)
		self.y_test = self.y_test - min(self.y_test)
		self.y_train_valid = np_utils.to_categorical(self.y_train_valid, self.num_classes)
		self.y_test = np_utils.to_categorical(self.y_test, self.num_classes)

		if (print_data):
			print ('Training/Valid data shape: {}'.format(self.X_train_valid.shape))
			print ('Test data shape: {}'.format(self.X_test.shape))
			print ('Training/Valid target shape: {}'.format(self.y_train_valid.shape))
			print ('Test target shape: {}'.format(self.y_test.shape))
			print ('Person train/valid shape: {}'.format(self.person_train_valid.shape))
			print ('Person test shape: {}'.format(self.person_test.shape))
		print("Done loading data")

	def construct_model(self, lr = 0.001, lr_decay = 0.99, lstm_outputs=256, 
					dropout_prob = 0.5, recurrent_dropout = 0.5,use_batchnorm = True, print_model = True):
		from keras.models import Sequential
		from keras.layers import LSTM, Dense, Dropout
		from keras import optimizers
		from keras.layers.normalization import BatchNormalization

		_, timesteps, data_dim = self.X_train_valid.shape
		optimizer = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=lr_decay)
		self.model = Sequential()
		self.model.add(LSTM(lstm_outputs, activation= 'tanh', recurrent_activation='hard_sigmoid', 
					   dropout=dropout_prob,  recurrent_dropout = recurrent_dropout, return_sequences=True,
					   input_shape=(timesteps, data_dim)))
		if (use_batchnorm):
			self.model.add(BatchNormalization())

		self.model.add(LSTM(lstm_outputs, activation= 'tanh', recurrent_activation='hard_sigmoid', 
					   dropout=dropout_prob,  recurrent_dropout = recurrent_dropout))
		if (use_batchnorm):
			self.model.add(BatchNormalization())

		self.model.add(Dense(64, activation='relu'))
		self.model.add(Dropout(dropout_prob))
		if (use_batchnorm):
			self.model.add(BatchNormalization())

		self.model.add(Dense(16, activation='relu'))
		self.model.add(Dropout(dropout_prob))
		if (use_batchnorm):
			self.model.add(BatchNormalization())

		self.model.add(Dense(self.num_classes, activation='softmax'))
		self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

		if (print_model):
			self.model.summary()
		with open(ARCHITECTURE_PATH, 'w') as f:
			f.write(self.model.to_json())

	def train(self, batch_size = 64, epochs = 1, validation_split = 0.1):
		print("Start Training...")
		from keras.callbacks import ModelCheckpoint
		checkpointer = ModelCheckpoint(filepath= WEIGHTS_PATH, verbose=1, save_best_only=True)
		self.history = self.model.fit(self.X_train_valid, self.y_train_valid, batch_size=batch_size, 
					epochs=epochs, verbose=1, callbacks=[checkpointer], validation_split = validation_split, shuffle=True)
		print("Training Done.")

	def load_model(self, print_model = False):
		# load the trained model
		from keras.models import model_from_json
		with open(ARCHITECTURE_PATH, 'r') as f:
			self.model = model_from_json(f.read())

		self.model.load_weights(WEIGHTS_PATH)
		if (print_model):
			self.model.summary()

	def output_test_scores(self):
		# Get predicted action for each signal in test set
		predictions = self.model.predict(self.X_test)
		# print out test accuracy
		test_accuracy = 100*np.sum(np.argmax(predictions,axis =1)==np.argmax(self.y_test, axis=1))/len(predictions)
		print('Test accuracy: %.4f%%' % test_accuracy)

def main():
	model = RNN(print_data = False)
	# model.construct_model()
	# model.train()
	model.load_model(False)
	model.output_test_scores()


if __name__ == '__main__':
	main()



