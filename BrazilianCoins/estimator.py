import cv2,os,random
import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical
#https://www.kaggle.com/gbonesso/deep-learning-cnn/data

tf.logging.set_verbosity(tf.logging.DEBUG)
label_dict,id_to_name1={},[]
base='/Users/ckl/dl/data/coins/single/'

def extract_coins(img,toSize=100):
	cimg=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	circles=cv2.HoughCircles(cimg, cv2.HOUGH_GRADIENT,2,60,param1=300, param2=30, minRadius=30, maxRadius=50)

	if circles is None:
		return None,None;

	hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	lower=np.array([0,0,0])
	higher=np.array([255,255,90])
	mask=cv2.blur(cv2.inRange(hsv,lower,higher),(8,8))

	radiuses,frames=[],[]

	for circle in circles[0]:
		cx=int(circle[0])
		cy=int(circle[1])

		if cx >= len(mask) or cy >= len(mask[cx]) or not mask[cx,cy]:
			continue #???

		radius=circle[2]+3
		radiuses.append(radius)
		x=int(cx-radius)
		y=int(cy-radius)

		if y<0:
			y=0
		if x<0:
			x=0

		resized=cv2.resize(img[y:int(y+2*radius), x:int(x+2*radius)],(toSize,toSize),interpolation=cv2.INTER_CUBIC)
		frames.append(resized)

	return np.array(frames),radiuses

def read_extract(path,p):
	img=cv2.imread(path+p,cv2.IMREAD_COLOR)

	ready,_=extract_coins(img)
	if ready is not None and len(ready)>0:
		return ready[0],p.split('_')[0]
	return None,None

def readFiles(path):
	cnt,idx=0,-1
	scaled_raw,scaled_labels,id_to_name2=[],[],{}
	for p in os.listdir(path):
		if p[-4:]=='.jpg':
			img,label=read_extract(path,p)
			if img is not None:
				scaled_raw.append(img)
				scaled_labels.append(label)
				id_to_name2[len(scaled_raw)-1]=p
	print 'read files end.'
	imgs,labels=np.array(scaled_raw),np.array(scaled_labels)
	return imgs,labels,id_to_name2

def formatLabels(labels):
	label_classes=set(labels)
	for v_i, v in enumerate(label_classes):
		label_dict[v]=v_i

	res=[]
	for label in labels:
		res.append(label_dict[label])
	
	return res

def buildEstimator(features,labels,mode):
	imgs=tf.reshape(features['x'],[-1,100,100,3])
	conv1=tf.layers.conv2d(# 100*100*3*32
		inputs=imgs,
		filters=32,
		kernel_size=[3,3],
		padding='same',
		activation=tf.nn.relu
	)
	pool1=tf.layers.max_pooling2d(inputs=conv1,pool_size=(2,2),data_format='channels_last',strides=(2,2))# 50*50*3*32

	conv2=tf.layers.conv2d(
		inputs=pool1,
		filters=32,
		kernel_size=[3,3],
		padding='same',
		activation=tf.nn.relu
	)
	pool2=tf.layers.max_pooling2d(inputs=conv2, pool_size=(2,2),strides=(2,2))# 25*25*3*32

	conv3=tf.layers.conv2d(
		inputs=pool2,
		filters=64,
		kernel_size=[3,3],
		padding='same',
		activation=tf.nn.relu
	)
	pool3=tf.layers.max_pooling2d(inputs=conv3,pool_size=(2,2),data_format='channels_last', strides=(2,2))# 25*25*3*64??
	
	pool3_flat=tf.reshape(pool3,[-1,12*12*64])
	print pool3_flat

	dense1=tf.layers.dense(inputs=pool3_flat,units=64)

	batch_norm1=tf.layers.batch_normalization(
		inputs=dense1,
		training=True
	)

	relu1=tf.nn.relu(batch_norm1)

	
	dropout=tf.layers.dropout(inputs=relu1, rate=0.75)
	logits=tf.layers.dense(inputs=dropout,units=5)

	print mode,mode==tf.estimator.ModeKeys.PREDICT

	if mode==tf.estimator.ModeKeys.PREDICT:
		predictions={
			'classes':tf.argmax(logits, axis=1),
			'probabilities':tf.nn.softmax(logits, name='softmax_tensor')
		}
		print predictions
		return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)
	
	if mode==tf.estimator.ModeKeys.TRAIN:
		optimizer=tf.train.AdamOptimizer(learning_rate=1e-4)

		logits=tf.nn.softmax(logits)
		loss=tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)

		accuracy=tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits,axis=1))
		tf.identity(accuracy[1], name='train_accuracy')
		tf.summary.scalar('train_accuracy',accuracy[1])
		
		return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN,loss=loss,
			train_op=optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
		)

def build2(scaled,y_binary):
	from keras import applications, optimizers, Input
	from keras.models import Sequential, Model
	from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
	from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
	from keras.utils.np_utils import to_categorical
	from keras.layers.normalization import BatchNormalization

	y_binary=to_categorical(y_binary)

	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(100, 100, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(data_format="channels_last", pool_size=(2, 2)))

	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu')) 
	model.add(MaxPooling2D(data_format="channels_last", pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(data_format="channels_last", pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(64))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.75))          # 0.5
	model.add(Dense(5))               # 5 is the number of classes
	model.add(Activation('softmax'))

	model.compile(
	    loss='categorical_crossentropy', 
	    optimizer='adam',              # adadelta
	    metrics=['accuracy']
	)

	model.fit(
		x=np.array(scaled),
		y=y_binary,
		epochs=10,
		validation_split=0.15,
		batch_size=500,
		verbose=1
	)

def show(img):
	cv2.imshow('xxx',img)
	cv2.waitKey()

def findLabel(id):
	return label_dict.keys()[list(label_dict.values()).index(id)]

def predict(classifier,img):
	pred_input_fn=tf.estimator.inputs.numpy_input_fn(
		x={'x':np.asarray([img],dtype=np.float32)},
		num_epochs=1,
		shuffle=False
	)

	predictions=list(classifier.predict(input_fn=pred_input_fn))
	predicted_classes = [p["classes"] for p in predictions]
	print predicted_classes
	return predictions

def init():
	classifier=tf.estimator.Estimator(
		model_fn=buildEstimator,
		model_dir='/Users/ckl/tmp/br-coins'
	)
	return classifier

def train(classifier):
	scaled_raw, scaled_labels,id_to_name2 =readFiles(base)

	labels=formatLabels(scaled_labels)

	print label_dict

	print 'labels: ',labels
	classifier=init()

	scaled=np.asarray(scaled_raw,dtype=np.float32)
	print len(labels)

	labels=np.asarray(labels,dtype=np.int32)

	train_input_fn=tf.estimator.inputs.numpy_input_fn(
		x={"x":scaled},
		y=labels,
		batch_size=500,
		num_epochs=10,
		shuffle=True
	)
	tensors_to_log = {'train_accuracy': 'train_accuracy'}
	logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=1)
	classifier.train(
		input_fn=train_input_fn,
		steps=100,
		hooks=[logging_hook]
	)

def next():
	while True:
		mlst=os.listdir(base)
		idx=random.randint(0,len(mlst)-1)

		img,label=read_extract(base,mlst[idx])
		if img is not None:
			return  img,label

if __name__=='__main__':
	
	x=estimator.init()
	estimator.train(x)

	for i in range(100):
		img,label=next()

		print predict(est,img)

		

		
