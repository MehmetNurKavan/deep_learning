# %% veri seti yÃ¼klenmesi and preprocessing
import  numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

#%% load data , amacÄ±mÄ±z gÃ¶rÃ¼ntÃ¼elri sÄ±kÄ±ÅtÄ±rmakolduÄundan, etiketlerie ihtiyacÄ±mÄ±z yok, o yÃ¼zden y_train, y_testi sildim "_" yaptik
(x_train, _), (x_test, _) = fashion_mnist.load_data()


#%% normalization
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

#%%
# gorsellestirme
plt.figure()
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_train[i], cmap="gray")
    plt.axis("off")
plt.show()

# %% vveriyi uzenlestir 28x28 boyutundaki goruntuleri 784 boyutunca bir vektrıe donsutur

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

'''
    otoencoders 2 boutlu veriyi giris oalrak alamiyor, tek boyutlu olamsı gerekiyor.
 x_train.shape = (60000, 28, 28)  # 60,000 tane 28x28'lik görüntü iken
 x_train.shape = (60000, 784)  # Her görüntü 28x28 = 784 boyutlu bir vektör oldu

 x_test.shape = (10000, 28, 28)  # 60,000 tane 28x28'lik görüntü iken
 x_test.shape = (10000, 784)  # Her görüntü 28x28 = 784 boyutlu bir vektör oldu 
'''

#%% encoder ve decoder mimariis olsuturma, autoendoerss= encoder + decoder , training

# autoendocer icin model parametrelerinin tanımlanmasi
input_dim = x_train.shape[1] # 784
encoding_dim = 64 # latten veriyi kuculttukten sorna elde ettigimiz parca boyutu

# encoder
input_image = Input(shape = (input_dim, )) # girdi boyutunu belirleme
encoded = Dense(256, activation = "relu")(input_image) # ilk gizli katman 256 noron
encoded = Dense(128, activation = "relu")(encoded) # ikinci gizli katmani 128 norın
encoded = Dense(encoding_dim, activation = "relu")(encoded) # sıkışıtırma katmanı, latent 

# decoder
decoded = Dense(128, activation = "relu")(encoded) # ilk genisletme katmani
decoded = Dense(256, activation = "relu")(decoded) # ikinci genisletme katmani
decoded = Dense(input_dim, activation = "sigmoid")(decoded) # sigmoit inputmuzuzu 0-1 arasinda sıkıştırıyor # compresses data between 0 and 1
 # ikinci parantezle baglama parantezi oluyor farkedersen, mesela altta decoded olmasina ragmen ikinci parantez encoded, encodedi decodeda baglamis cünküü

try:
    autoencoder = load_model('autoencoder_model.h5')
    print("Model başarıyla yüklendi.")
except:
    print("Model bulunamadı, sıfırdan eğitim yapılacak.")

    # autoencoders
    autoencoder = Model(input_image, decoded) # giristen ciktiya tum yapiyi tanimliyoruız
    
    # compile
    autoencoder.compile (optimizer = Adam(), loss = "binary_crossentropy")
    
    # training
    history = autoencoder.fit(
        x_train, # girdi ve hedef(target) aynı olamalı (otonom ogrenme)
        x_train,  # amacimiz zaten orijican görüntüyü oluşturmak olduğu için orijinal olan gene x_trraini verdik ki yukarda y kısmlarinı yapmayip "_" yazmistik kullanmicaz diye
        epochs = 50,
        batch_size = 64,
        shuffle  = True, # egitim verilerini karistir
        validation_data = (x_test, x_test),
        verbose = 1,
        )
    
    # Modeli kaydet
    autoencoder.save('autoencoder_model.h5')
    print("Model kaydedildi.")

'''
Basit (Sığ) Model:
    Encoder Katmanları: 32 → 16(encoding_dim)
    Decoder Katmanları: 16 → 32 → 784(input_dim)
    sonuc: loss: 0.2862 - val_loss: 0.2887
Orta Seviye Model:
    Encoder Katmanları: 256 → 128 → 64(encoding_dim)
    Decoder Katmanları: 64 → 128 → 256 → 784(input_dim)
    sonuc: loss: 0.2619 - val_loss: 0.2647
Karmaşık Seviye Model:
    Encoder Katmanları: 512 → 256 → 128 → 64(encoding_dim)
    Decoder Katmanları: 64 → 128 → 256 → 512 → 784(input_dim)
    sonuc: loss: 0.2624 - val_loss: 0.2658

incelenince orta seviye modeli kullanmak mantıklı geliyor

'''

# %% model test and evalution: PSNR degerlendirme metrik

# encoder
encoder = Model(input_image, encoded)

# decoder
encoded_input = Input(shape = (encoding_dim,))
decoder_layer1 = autoencoder.layers[-3](encoded_input)
decoder_layer2 = autoencoder.layers[-2](decoder_layer1)
decoder_output = autoencoder.layers[-1](decoder_layer2)

decoder = Model(encoded_input, decoder_output)

# test verisi ile encoder ile sıkıştırma ve decoder ile yeniden yapılandırma
encoded_images = encoder.predict(x_test)  # latent temsili haline getir
decoded_images = decoder.predict(encoded_images)  # latent temsilden orijinal formata geri çevir

# orijinal ve yeniden yapılandırılmış (decoded_images) görüntülerini görselleştir
n = 10  # 10 örnek

plt.figure(figsize = (15,5))
for i in range(n):
    # orijinal görüntü
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap = "gray")
    ax.get_xaxis().set_visible(False)  # x eksenini gizle
    ax.get_yaxis().set_visible(False)  # y eksenini gizle
    
    # decoded_image / yeniden yapılandırılmış görüntü
    ax = plt.subplot(2, n, i + 1 + n)  # ikinci satırdaki subplot
    plt.imshow(decoded_images[i].reshape(28, 28), cmap = "gray")
    ax.get_xaxis().set_visible(False)  # x eksenini gizle
    ax.get_yaxis().set_visible(False)  # y eksenini gizle

plt.show()

# peak- snr : PSNR
def calculate_psnr(original, reconstructed):
    # her iki goruntu arasindaki psnr i hesapla
    mse = np.mean((original - reconstructed)**2) # mean square error
    
    if mse == 0:
        return float("inf") # ilk goruntu tamamne ayni ise psnr =sonsuz
    
    max_pixek = 1.0
    psnr = 20 * np.log10(max_pixek / np.sqrt(mse))
    return psnr


# test verileri icin psnr hesaplama
psnr_score = []

# ilk 100 ornek icin yapalim
for i in range(100):
    original_img = x_test[i]
    reconstructed_img = decoded_images[i]
    score = calculate_psnr(original_img, reconstructed_img)
    psnr_score.append(score)
    
average_psnr = np.mean(psnr_score) 
print(f"average_psnr: {average_psnr}")   
