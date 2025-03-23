# 🧠 Deep Learning Repository

Bu depo, farklı **derin öğrenme** tekniklerini içeren projeleri barındırmaktadır. Derin Öğrenme alanıı öğrendiğim süreçte yaptığım çalışmaları içeren koleksiyonudur.  Her klasör belirli bir derin öğrenme modeline odaklanmaktadır. 📚
---

## 🛠 Kullanılan Teknolojiler
Bu projede aşağıdaki Python kütüphaneleri ve araçları kullanılmıştır:

- **Pandas** → Veri işleme ve manipülasyon
- **NumPy** → Matematiksel işlemler
- **Matplotlib & Seaborn** → Veri görselleştirme
- **TensorFlow & Keras** → Derin Öğrenme Algoritmaları
---

## 📂 İçindekiler

### 🔵 101_ANN | **Yapay Sinir Ağları (Artificial Neural Networks)**
Yapay Sinir Ağları (ANN), insan beyninin çalışma prensibini taklit eden algoritmalardır. Bu model, basit bir ileri beslemeli yapı kullanarak veri sınıflandırma ve regresyon problemlerinde etkili sonuçlar elde eder.

📌 **Dosyalar:**
- `1_mnist_model.h5` → MNIST veri seti için eğitilmiş ANN modeli.
- `1_mnist.ipynb` → MNIST el yazısı rakamları sınıflandırma modeli.
- `2_cifar10_ann_model.keras` → CIFAR-10 veri seti için eğitilmiş ANN modeli.
- `2_cifar10_ann.py` → CIFAR-10 ile ANN eğitimi için Python betiği.

---

### 🟠 201_CNN | **Konvolüsyonel Sinir Ağları (Convolutional Neural Networks)**
CNN'ler, görüntü işleme ve sınıflandırma için kullanılan derin öğrenme modelleridir. Görsellerdeki örüntüleri ve özellikleri tespit etmek için konvolüsyon katmanlarını kullanırlar.

📌 **Dosyalar:**
- `garbage.ipynb` → Çöp sınıflandırma için CNN modeli.

---

### 🟡 301_RNN | **Tekrarlayan Sinir Ağları (Recurrent Neural Networks)**
RNN'ler, sıralı verilerle çalışmak için geliştirilmiş bir sinir ağı türüdür. Zaman serisi tahmini, dil modeli ve metin analizi gibi alanlarda yaygın olarak kullanılır.

📌 **Dosyalar:**
- `international-airline-passe...` → Uçak yolcu verileri ile zaman serisi analizi.
- `mymodel.keras` → Eğitilmiş RNN modeli.
- `RNN.ipynb` → Zaman serisi tahmini ve analiz modeli.

---

### 🟢 401_LSTM | **Uzun Kısa Süreli Bellek (Long Short-Term Memory)**
LSTM, RNN'lerin gelişmiş bir versiyonudur ve uzun vadeli bağımlılıkları daha iyi öğrenebilir. Duygu analizi, hisse senedi tahmini ve doğal dil işleme gibi alanlarda kullanılır.

📌 **Dosyalar:**
- `1_LSTM.pynb.ipynb` → LSTM modeli kullanılrak Tesla hisse senedi ile zaman serisi tahmini.
- `1_mymodel.keras` → Tesla hisse senedi eğitilmiş LSTM modeli.
- `1_TSLA.csv` → Tesla hisse senedi fiyatları veri seti.
- `2_film_yorum.py` → LSTM ile film yorum analizi.
- `2_loss_acc.png` → LSTM ile film yorum analizi Modelin eğitim sürecindeki doğruluk ve kayıp grafiği.
- `2_LSTM_yorum_duygu_analiz_model.h5` → LSTM ile duygu analizi projesi model dosyası.

---

### 🔴 501_gan | **Üretici Karşıt Ağlar (Generative Adversarial Networks)**
GAN'ler, sahte veri üretmek için kullanılan yapay zeka modelleridir. İki ağ (üretici ve ayırt edici) birbirine karşı yarışarak öğrenir.

📌 **Dosyalar:**
- `mnist_gan.ipynb` → MNIST veri seti ile GAN uygulaması.

---

### 🟣 601_transform | **Dönüştürücü Modeller (Transformers)**
Transformers, doğal dil işleme (NLP) alanında devrim yaratan modellerdir. Chatbotlar, çeviri sistemleri ve metin özetleme gibi birçok alanda kullanılır.

📌 **Dosyalar:**
- `output.png` → Transformer modeli çıktı görseli.
- `yorum_duygu_analiz.ipynb` → Transformer ile duygu analizi çalışması.

---

### ⚪ 701_autoencoder | **Otoenkoderler (Autoencoders)**
Otoenkoderler, verileri sıkıştırmak ve önemli özellikleri çıkarmak için kullanılan sinir ağlarıdır. Gürültü giderme, anomalileri tespit etme ve özellik çıkarımı gibi alanlarda kullanılır.

📌 **Dosyalar:**
- `autoencoder_model.h5` → Eğitilmiş otoenkoder modeli.
- `moda.py` → Otoenkoder ile moda veri seti çalışması.

---

### 🟤 801_transfer_learning | **Transfer Öğrenme (Transfer Learning)**
Transfer öğrenme, önceden eğitilmiş bir modeli yeni bir veri setine uyarlayarak eğitim sürecini hızlandırmaya yardımcı olur.

* Bu proje, EfficientNetB0 modelini kullanarak katarakt tespiti yapmayı amaçlayan bir görüntü sınıflandırma modelidir.
  https://github.com/MehmetNurKavan/cataract_detection
* Bu proje, MobileNetV2 modelini kullanarak akciğer çökmesi tespiti yapmayı amaçlayan bir görüntü sınıflandırma modelidir.
  https://github.com/MehmetNurKavan/lung_collapse

---
