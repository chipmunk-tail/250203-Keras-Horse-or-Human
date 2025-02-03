
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten,Conv2D, MaxPool2D, Activation, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


# 가공해서 저장한 이미지를 불러오기
X_train = np.load('./Trains/binary_X_train.npy', allow_pickle = True)
print(X_train.shape)
X_test = np.load('./Trains/binary_X_test.npy', allow_pickle = True)
print(X_test.shape)
Y_train = np.load('./Trains/binary_Y_train.npy', allow_pickle = True)
print(Y_train.shape)
Y_test = np.load('./Trains/binary_Y_test.npy', allow_pickle = True)
print(Y_test.shape)

# 모델생성
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), input_shape = (150, 150, 3),
                 padding = 'same', activation = 'relu'))            # input_dim = 64 x 64, RGB(3색)
model.add(MaxPool2D(pool_size = (2, 2)))                            # MaxPool2D는 Conv2D랑 거의 세트이다.
model.add(Conv2D(32, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))            # 커널 사이즈는 3 x 3 (필터)
model.add(MaxPool2D(pool_size = (2, 2)))                            # 패딩을 이용해서 데이터 손실이 없이 사용
model.add(Conv2D(64, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2, 2)))

model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation = 'sigmoid'))                         # 개냐 아니냐 이중 분류기 = 시그모이드
model.summary()

model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam', metrics = ['accuracy'])

# 과적합이 오면 멈춘다 = 'val_accuracy'가 patience = 7회 학습동안 정확도가 올라가지 않으면 멈춘다.
# early_stopping = EarlyStopping(monitor = 'val_accuracy', patience = 7)
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 7)
fit_hist = model.fit(X_train, Y_train, batch_size = 64, epochs = 100,
                     validation_split = 0.15, callbacks = [early_stopping])

score = model.evaluate(X_test, Y_test)
print('Evaluation loss', score[0])
print('Ecaluation accuracy :', score[1])

# 실행 전 모델을 저장할 폴더 modles를 만들어야 한다
model.save('./Models/Horse_and_Human_binary_classfication_{}.h5'.format(score[1]))
plt.plot(fit_hist.history['loss'], label = 'loss')
plt.plot(fit_hist.history['val_loss'], label = 'alv_loss')
plt.legend()
plt.show()

plt.plot(fit_hist.history['accuracy'], label = 'accuracy')
plt.plot(fit_hist.history['val_accuracy'], label = 'val_accuracy')
plt.legend()
plt.show()


