# %%veri setinin yüklenilmesi

# veri setindeki yorumlarin olumlu mu olumsuz mu onu siniflandiricaz

import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from keras.models import load_model


# veri seti yukelme 
max_features = 10000 # en cok kullanılan kelimeler
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_features)

# veriyi padding islemi ile aynı uzunluga  getirme
maxlen = 100 # her yorum uzunlugu 100 kelime ile sınırlı
x_train = pad_sequences(x_train, maxlen = maxlen)
x_test = pad_sequences(x_test, maxlen = maxlen)

# %% model create, compile ve train

# model olusturma
def build_lstm_model():
    model = Sequential()
    model.add(Embedding(input_dim =  max_features, output_dim = 64, input_length = maxlen))
    model.add(LSTM(units = 8))
    model.add(Dropout(0.8))
    model.add(Dense(1, activation = "sigmoid")) # iki sinif diye isigmoit, çok olsa softmax kullanirdik    
    
    # derleme
    model.compile(optimizer = Adam(learning_rate = 0.0005),
                  loss = "binary_crossentropy",
                  metrics = ["accuracy"]
                  )
    
    return model

model = build_lstm_model()
model.summary()

# early stopping
# eger validation accuracy 3 kez ardı ardina degismiyorsa artık durdur dityoruz, en yiisini dondur bize diyoruz
early_stopping = EarlyStopping(monitor = "val_accuracy", patience = 3, restore_best_weights = True)

# training
history = model.fit(x_train, y_train, 
                    epochs = 10, 
                    batch_size = 16, 
                    validation_split= 0.2, 
                    callbacks = [early_stopping],
                    )
# loss: 0.1814 - accuracy: 0.9554 - val_loss: 0.4385 - val_accuracy: 0.8326
# bu degerleri incelerken modelin ezber yaptıını anlıyoruz cunku accuracy ile val_Accuracy arasidna yuzde11fark var
# modelin arkmasikligibni arttırısak ezberleme gidebiirlir, dropoutu artturma, unitsi ve output_dimi dusurmek

# %% model test ve evaluuate

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss : {loss}, test accuracy: {accuracy}")

plt.figure()
# loss
plt.subplot(1,2,1)
plt.plot(history.history["loss"], label = "training loss")
plt.plot(history.history["val_loss"], label = "validation loss")
plt.title("loss")
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend()
plt.grid(True)

# accracy
plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label = "training accuracy")
plt.plot(history.history["val_accuracy"], label = "validation accuracy")
plt.title("accuracy")
plt.xlabel("Epochs")
plt.ylabel("accuracy")
plt.legend()
plt.grid(True)

plt.show()

# %% model kaydetme

model.save('LSTM_yorum_duygu_analiz_model.h5')