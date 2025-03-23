# %% Veri Seti Hazirlama
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.models import load_model
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# veri seti yukle
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# f5 e tıklayinca tüm bo k çalişiyor
# f9 a basinca seçili ksıım çalişyor

plt.figure(figsize = (10, 5))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(x_train[i])
    plt.title(f"index: {i}, label: {y_train[i][0]}")
    plt.axis("off")
plt.show()

# normalization 0-1 arasinda scale etme
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# x_test ve x_traine tıklayinca hata veryor o yüzden konsolda a= x_train[0] yapipi a ile gbakiyoruz şimdiliik incelemek itersek

# one hot encoding
y_train = to_categorical(y_train, 10) # 10 = sinif sayisi
y_test = to_categorical(y_test, 10)


# %% ANN modelin olusturulamsi ve derlenilmesi
model = Sequential()

model.add(Flatten(input_shape = (32,32,3))) # 3D -> 1D

model.add(Dense(512, activation='relu')) # first layer
model.add(Dense(256, activation='tanh')) # second layer

model.add(Dense(10, activation="softmax")) # output layer (birden fazla sinifa ayrildigindan softmax secitk)

model.summary()


# modelin derlenilemsi
model.compile(optimizer = "adam",
              loss = "categorical_crossentropy",
              metrics = ["accuracy"],
              )

# %% callvack fonksiyonlarin tanitilmasi ve modelin gegitilmesi

# monitor: dogrulama setindeki(val) kaybi(loss) izler
# patience: 5 epoch boyunca val loss degismiyorsa erken durdurma yapalim

early_stopping = EarlyStopping(monitor = "val_loss", patience = 5, restore_best_weights = True)

# model chackpoint: en iyi modelin agirliklariin kaydeder
checkpoint =  ModelCheckpoint("ann_best_model.h5", monitor = "val_loss", save_best_only = True)
 
# model training 10 epcohs, batch size 60, val set %20

history = model.fit(x_train, 
                    y_train,
                    epochs = 10, 
                    batch_size = 60, # veri seti 60 ar parcalar ile egitilecek
                    validation_split = 0.2, # veri setinin yuzde 20 si dogrulama olarka kullanilcak
                    callbacks = [early_stopping, checkpoint])


# %% modelin test edilemsi ve performansinin incelenmesi 

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test loss: {test_loss}, test_acc: {test_acc}")

# trainnig and vlaidaiton accuracy visualization
plt.figure()
plt.plot(history.history["accuracy"], marker = "o", label = "Training Accuracy")
plt.plot(history.history["val_accuracy"], marker = "o", label = "Validation Accuracy")
plt.title("ANN Accuracy on CIFAT-10 Dataset")
plt.ylabel("Epochs")
plt.xlabel("Accuracy")
plt.legend() # yazilairn gorsel uzerine ekliyoruz
plt.grid(True) # izgara ekliyoruz
plt.show()

# training and validation loss visualization

plt.figure()
plt.plot(history.history["loss"], marker = "o", label = "Training Loss")
plt.plot(history.history["val_loss"], marker = "o", label = "Validation Loss")
plt.title("ANN Loss on CIFAR-10 Dataset")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


# model kaydetme 
model.save("ann_cifar10_model.keras")

# load model
loaded_model = load_model("ann_cifar10_model.keras")
test_loss, test_acc = loaded_model.evaluate(x_test, y_test)
print(f"Test loss: {test_loss}, test_acc: {test_acc}")

