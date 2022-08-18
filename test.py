import numpy as np
from tensorflow.keras import Model, Input
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Dropout, Embedding
from tensorflow.keras.preprocessing import sequence
from tcn import TCN
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.losses import MeanSquaredError

i = Input(shape = (200,7))
x = TCN(nb_filters = 64, kernel_size=6, dilations=[1,2,4,8,16])(i)
x = Dropout(0.5)(x)
x = Dense(1, activation = 'leaky_relu')(x)
model = Model(inputs=[i], outputs=[x])
model.compile(optimizer='adam', loss='mse', metrics =['MeanSquaredError'])
model.summary()

c1=np.load('D:\大四上学期\毕设\Tool Wear RUL based on ResNet\Data\c1_downsample_norm_trans.npy')
c4=np.load('D:\大四上学期\毕设\Tool Wear RUL based on ResNet\Data\c4_downsample_norm_trans.npy')
c6=np.load('D:\大四上学期\毕设\Tool Wear RUL based on ResNet\Data\c6_downsample_norm_trans.npy')

wear_c1=np.load('D:\大四上学期\毕设\Tool Wear RUL based on ResNet\Data\wear_c1.npy')
wear_c4=np.load('D:\大四上学期\毕设\Tool Wear RUL based on ResNet\Data\wear_c4.npy')
wear_c6=np.load('D:\大四上学期\毕设\Tool Wear RUL based on ResNet\Data\wear_c6.npy')

x=np.concatenate([c1,c4,c6],axis=0)
y=np.concatenate([wear_c1,wear_c4,wear_c6],axis=0)
y=np.reshape(y,[y.shape[0],1])

x_temporary=np.zeros([945,201,7])
for i in range(945):
    x_temporary[i,200,0]=y[i,0]
    x_temporary[i, 0:200, :] = x[i,:,:]
np.random.shuffle(x_temporary)
shuffle_train1=x_temporary[0:315]
shuffle_train2=x_temporary[315:630]
shuffle_train3=x_temporary[630:945]
def sort_dataset(dataset):
    for j in range(315):
        for i in range(315 - j - 1):
            if dataset[i, 200, 0] > dataset[i + 1, 200, 0]:
                dataset[[i, i + 1], :, :] = dataset[[i + 1, i], :, :]
    return dataset
shuffle_train1=sort_dataset(shuffle_train1)
shuffle_train2=sort_dataset(shuffle_train2)
shuffle_train3=sort_dataset(shuffle_train3)
sorted_train=np.concatenate([shuffle_train1,shuffle_train2],axis=0)

# plt.plot(shuffle_train1[0:315,200,0])
x_train=np.zeros([630,200,7])
y_train=np.zeros([630,1])
x_test=np.zeros([315,200,7])
y_test=np.zeros([315,1])
for i in range(630):
    x_train[i,:,:]=sorted_train[i,0:200,:]
    y_train[i,0]=sorted_train[i,200,0]
for i in range(315):
    x_test[i, :, :] = shuffle_train3[i, 0:200, :]
    y_test[i, 0] = shuffle_train3[i, 200, 0]

# history = model.fit(x_train, y_train, batch_size = 315, epochs = 300, validation_data = (x_train, y_train))
model=tf.keras.models.load_model('model/loss_7')
result = model.evaluate(x_test, y_test)
#  172.6868 210.9194 234.7159
#  125:300  55:283   70:245
#  7:266
r=model.predict(x_train[0:315])
mse=MeanSquaredError()
print(f"rmse is: {np.sqrt(mse(r[7:266],y_train[7:266]))}")
plt.plot(r[7:266])
plt.plot(y_train[7:266])


