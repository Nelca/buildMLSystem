from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()

model.add(Dense(units=64, input_dim=100))
model.add(Activation('relu'))
model.add(Dense(units=64))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batchsize=32)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

