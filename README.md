## Kırmızı Şarap Kalite Tahmini
Bu proje, UCI Machine Learning Repository’de yer alan kırmızı şarap veri seti kullanılarak şarap kalitesinin tahmin edilmesini amaçlamaktadır.
Veri seti, farklı kimyasal özellikler (asitlik, şeker, pH, alkol oranı gibi) ve bunların kalite üzerindeki etkisini içermektedir. 
Projede, şarap kalitesi 3 sınıfa ayrılarak (kötü, orta, iyi) sınıflandırma yapılmıştır.
Model LightGBM algoritması ile eğitilmiş ve sonuçların test edilmesi için Flask tabanlı bir web arayüzü geliştirilmiştir.

## Veri Seti Hakkında
* Toplam örnek sayısı: 1599
* Bağımsız Değişken Sayısı: 11
* Hedef Değişken: quality (0–10 arasında tam sayılar)
* Dosya Formatı: CSV


* Değişkenler:
  * fixed acidity (sabit asitlik)
  * volatile acidity (uçucu asitlik)
  * citric acid (sitrik asit)
  * residual sugar (kalıntı şeker)
  * chlorides (klorür)
  * free sulfur dioxide (serbest SO2)
  * total sulfur dioxide (toplam SO2)
  * density (yoğunluk)
  * pH
  * sulphates (sülfatlar)
  * alcohol (alkol oranı)

* Hedef: quality (3–8 arasında puan, 3 sınıfa indirgendi)


## Proje Detayları
* UCI kırmızı şarap veri seti kullanımı
* Şarap kalitesini kötü, orta ve iyi olarak sınıflandırma
* LightGBM tabanlı model eğitimi
* Flask ile web tabanlı tahmin arayüzü
