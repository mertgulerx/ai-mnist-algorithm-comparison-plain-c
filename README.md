# MNIST Veri Seti ile Temel Makine Öğrenimi Algoritmalarının Karşılaştırılması

## İçindekiler
1. [Giriş](#giriş)
   - [Ödevin Amacı](#ödevin-amacı)
2. [Yöntem](#yöntem)
   - [Kullanılan Veri Seti](#kullanılan-veri-seti)
   - [Veri İşleme](#veri-i̇şleme)
   - [Ağ Yapısı](#ağ-yapısı)
   - [Kullanılan Diller ve Kütüphaneler](#kullanılan-diller-ve-kütüphaneler)
   - [Kayıp Fonksiyonu ve Doğruluk Hesabı](#kayıp-fonksiyonu-ve-doğruluk-hesabı)
   - [Algoritmalar](#algoritmalar)
3. [Deneyler ve Sonuçlar](#deneyler-ve-sonuçlar)
   - [Eğitim Sonuçları](#eğitim-sonuçları)
   - [Batch Boyutu Karşılaştırması](#batch-boyutu-karşılaştırması)
   - [Test Sonuçları](#test-sonuçları)
4. [Tartışma](#tartışma)
5. [Sonuç](#sonuç)
6. [Kaynakça](#kaynakça)

---

## Giriş

Günümüzde yapay zeka, özellikle **derin öğrenme** ve **makine öğrenimi** alanlarında büyük bir ilerleme kaydetmiştir. Bu proje, yaygın olarak kullanılan **MNIST veri seti** ile temel makine öğrenimi algoritmalarını karşılaştırarak, hangi optimizasyon yönteminin en verimli sonuçları verdiğini analiz etmeyi amaçlamaktadır. 

MNIST veri seti, **0-9 arasındaki el yazısı rakamlarının** bulunduğu geniş çaplı bir veri setidir ve genellikle **görüntü işleme, sinir ağları ve model optimizasyonu** alanlarında kullanılan bir benchmark olarak kabul edilir. Bu proje kapsamında **Gradient Descent (GD), Stochastic Gradient Descent (SGD) ve ADAM** gibi üç popüler optimizasyon algoritması test edilerek, farklı yöntemlerin doğruluk, hız ve stabilite açısından performansları değerlendirilmiştir.

Makine öğrenimi modellerinin başarısı, yalnızca kullanılan algoritmalara değil, aynı zamanda **veri ön işleme, modelin ağırlıklarının başlatılması ve optimizasyon süreçlerinin etkinliğine** bağlıdır. Bu nedenle, proje sürecinde veri seti dikkatlice işlenmiş, model ağırlıkları rastgele ancak belirli bir aralıkta atanmış ve her algoritma için farklı öğrenme oranları ve batch boyutları test edilmiştir.

Ayrıca, optimizasyon algoritmalarının **eğitim süresi**, **doğruluk oranı** ve **hata fonksiyonu değerleri** gibi farklı metrikler üzerinde nasıl bir etkiye sahip olduğu incelenmiş ve sonuçlar detaylı grafikler ile görselleştirilmiştir.

### Ödevin Amacı

Bu çalışmanın temel amacı, **farklı optimizasyon algoritmalarının eğitim süreçlerinde nasıl performans gösterdiğini analiz etmek ve hangi algoritmanın en verimli olduğunu belirlemektir.**

Özellikle aşağıdaki sorulara yanıt aranmıştır:
- Hangi optimizasyon algoritması daha hızlı yakınsar?
- Batch boyutunun doğruluk ve kayıp değerleri üzerindeki etkisi nedir?
- Büyük veri setleri için en uygun yöntem hangisidir?
- GD, SGD ve ADAM algoritmaları, eğitim süreci boyunca nasıl farklılık göstermektedir?
- Ağırlık başlatma yöntemleri optimizasyon sürecine nasıl etki eder?
- Öğrenme oranı değişiklikleri algoritmaların başarımını nasıl etkiler?

Bu analizler, özellikle **derin öğrenme ve yapay sinir ağları** gibi daha karmaşık modellerin geliştirilmesinde doğru optimizasyon yönteminin seçilmesine yardımcı olabilir.

Bu projenin sonunda, her üç algoritmanın **avantajları ve dezavantajları** detaylı şekilde değerlendirilerek, belirli bir problem için hangi yöntemin daha uygun olabileceği konusunda öneriler sunulacaktır.

---

## Yöntem

### Kullanılan Veri Seti
MNIST veri seti, el yazısı rakamları içeren popüler bir veri setidir. **28x28 piksel boyutlarında, gri tonlamalı toplam 42.000 görüntü içermektedir.** Bu veri seti, **el yazısı rakam sınıflandırma** problemlerinde sıkça kullanılan bir benchmark veri setidir.

### Veri İşleme
- **Kaynak:** `.csv` formatında bulunan veri, her bir satırda 785 değer içermektedir (ilk sütun etiket, geri kalanlar piksel değerleri).
- **Önişleme:** Tüm piksel değerleri **0 ile 1** arasına normalize edilmiştir.
- **Format:** 28x28 görüntüler **vektör** formuna dönüştürülmüş ve bias terimi eklenmiştir.
- **Veri Ayrımı:** Veri seti eğitim ve test seti olarak ikiye ayrılmıştır. Eğitim seti 35.000, test seti 7.000 örnek içermektedir.

### Ağ Yapısı
Bu modelde **yalnızca giriş ve çıkış katmanları** bulunmaktadır:
- **Giriş katmanı:** 785 nöron (784 piksel + bias)
- **Çıkış katmanı:** 2 nöron (0 ve 1 sınıfları için)
- **Aktivasyon fonksiyonu:** `tanh` kullanılmıştır, çünkü -1 ile 1 aralığında simetrik bir çıkış sağlar ve öğrenme sürecini hızlandırabilir.

### Kullanılan Diller ve Kütüphaneler
- **C dili:** Model eğitimi ve ağırlık güncellemeleri için
- **Python:** Sonuçların görselleştirilmesi için (`matplotlib`, `pandas`, `scikit-learn`)

### Kayıp Fonksiyonu ve Doğruluk Hesabı
- **Mean Squared Error (MSE)** kullanılarak hata hesaplanmıştır.
- **GD için:** Hata oranı, toplam eğitim örneği sayısına bölünerek hesaplanmıştır.
- **SGD ve ADAM için:** Her iterasyonda **batch boyutu baz alınarak** hata hesaplanmıştır.

### Algoritmalar
- **Gradient Descent (GD):** Tüm eğitim verisini kullanarak güncelleme yapan klasik optimizasyon algoritmasıdır.
- **Stochastic Gradient Descent (SGD):** Rastgele seçilen tek bir örnek veya küçük batch'ler ile güncelleme yaparak daha hızlı bir öğrenme sağlar.
- **ADAM:** Öğrenme hızını dinamik olarak ayarlayan, momentum ve adaptif öğrenme oranlarını birleştiren bir algoritmadır.

---

## Deneyler ve Sonuçlar

Deney sonuçları grafikler ile görselleştirilmiştir.

### Eğitim Sonuçları
- **SGD ve ADAM hızlı yakınsarken, GD daha fazla güncellemeye ihtiyaç duymaktadır.**
- **Ağırlık aralıklarının büyümesiyle performans düşmüştür.**
- **SGD ve ADAM algoritmaları dalgalanmalara sahipken, GD stabil ancak yavaş ilerlemiştir.**
- **Ağırlık başlatma stratejileri, öğrenme sürecine doğrudan etki etmiştir.**

### Batch Boyutu Karşılaştırması
- **Batch boyutu arttıkça modelin kararlılığı artmıştır.**
- **SGD ve ADAM için en verimli batch boyutu 128 olarak belirlenmiştir.**
- **Küçük batch boyutlarında daha fazla dalgalanma gözlemlenmiştir.**
- **Büyük batch boyutları doğruluk oranını artırsa da, işlem süresini uzatmıştır.**

### Test Sonuçları
- **SGD ve ADAM doğruluk açısından yakın sonuçlar vermiştir.**
- **ADAM daha stabil sonuçlar sağlarken, SGD daha hızlı öğrenme göstermiştir.**
- **GD, diğer algoritmalara kıyasla daha düşük doğruluk seviyelerine ulaşmıştır.**
- **Batch boyutları test sürecinde de doğruluk ve hata oranlarını etkilemiştir.**

#### Doğruluk Sonuçları (Test Seti)
- **GD:** %97.2
- **SGD:** %99.02
- **ADAM:** %99.11

---

## Tartışma

### Algoritma Performansı
- **GD**, yavaş ve büyük veri setlerine uygun değildir.
- **SGD**, en hızlı algoritmadır ve büyük veri setlerinde daha verimli çalışır.
- **ADAM**, stabilite ve doğruluk açısından genellikle en iyi sonuçları vermiştir.
- **Öğrenme oranlarının değişimi her üç algoritmanın performansını farklı seviyelerde etkilemiştir.**

### Model Sınırlamaları
- **Gizli katman içermemesi**, modelin karmaşık ilişkileri öğrenmesini sınırlandırmıştır.
- **MNIST gibi nispeten basit bir veri seti kullanılmıştır, dolayısıyla daha kompleks veri setlerinde farklı sonuçlar elde edilebilir.**
- **Modelin performansı, kullanılan ağırlık başlatma yöntemi ve optimizasyon tekniklerine doğrudan bağlıdır.**
- **Aktivasyon fonksiyonları ve optimizasyon yöntemleri daha ileri düzey mimarilerde farklılık gösterebilir.**

---

## Sonuç

MNIST veri seti üzerinde yapılan karşılaştırma sonucunda:

| **Özellik**  | **GD** | **SGD** | **ADAM** |
|-------------|--------|--------|--------|
| Stabilite  | ✅ | ❌ | ✅ |
| Doğruluk  | ❌ | ✅ | ✅ |
| Hız | ❌ | ✅ | ✅ |

- **SGD**, yüksek hız ve doğruluk oranı ile büyük veri setleri için daha uygundur.
- **ADAM**, stabil doğruluk oranı ve orta seviye hız avantajı ile hassas veri setlerinde tercih edilebilir.
- **GD**, küçük veri setleri için en stabil algoritmadır, ancak yavaş olması nedeniyle büyük veri setlerinde önerilmez.
- **Batch boyutları ve öğrenme oranı seçimleri, modelin başarımını büyük ölçüde etkilemektedir.**

---

## Kaynakça
- [MNIST Veri Seti](https://www.kaggle.com/datasets/bhavikjikadara/handwritten-digit-recognition)



