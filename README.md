# Voice DeepFake Detection - Precision Model Performance Notes

Bu proje, ses verilerinde gercek/fake ayrimi yapmak icin egitilen segment tabanli CNN modelinin performansini olcmeyi amaçlamaktadır.

## Kullanilan Model

Degerlendirmede kullanilan en iyi model:

```text
checkpoints/precision_best.pth
```

Model, ses dosyasini tek parca olarak degerlendirmek yerine dosyayi 2 saniyelik segmentlere ayirir. Her dosya icin 8 segment uretilir. Model hem segment seviyesinde fake olasiligi uretir hem de bu segment bilgilerini birlestirerek dosya seviyesinde nihai karar verir.

Sinif etiketleri:

```text
0 = Real / Bonafide
1 = Fake / Spoof
```

## Performans Degerlendirmesi

Son performans kosusunda 100 dosya kullanilmistir:

```text
Real dosya: 14
Fake dosya: 86
Toplam dosya: 100
Her dosya icin segment: 8
Toplam segment: 800
```

Kullanilan threshold; traning de optimal threshold olçülmüştür:

```text
threshold = 0.255
```

Bu deger test sonucuna bakilarak elle secilmemistir. Egitim/validasyon surecinde en iyi checkpoint ile birlikte kaydedilen karar esigi kullanilmistir.

## Dosya Seviyesi Sonuclari

Dosya seviyesinde confusion matrix:

```text
                 Pred Real   Pred Fake
True Real           14           0
True Fake            0          86
```

Dosya seviyesinde metrikler:

```text
Accuracy  = 1.000
Precision = 1.000
Recall    = 1.000
F1-score  = 1.000
```

Yorum:

Model dosya seviyesinde bu test kosulunda tum ornekleri dogru siniflandirmistir. Ancak veri dagilimi fake sinifi lehine dengesiz oldugu icin bu sonuc yalnizca accuracy ile yorumlanmamalidir. Bu nedenle segment seviyesi, confusion matrix, precision, recall ve F1-score birlikte raporlanmistir.

## Segment Seviyesi Sonuclari

Segment seviyesinde confusion matrix:

```text
                 Pred Real   Pred Fake
True Real           96          16
True Fake           22         666
```

Segment seviyesinde metrikler:

```text
Accuracy  = 0.9525
Precision = 0.9765
Recall    = 0.9680
F1-score  = 0.9723
```

Yorum:

Segment seviyesinde bazi hatalar vardir. Real segmentlerin 16 tanesi fake olarak, fake segmentlerin 22 tanesi real olarak tahmin edilmistir. Buna ragmen dosya seviyesinde karar dogru cikmistir; cunku dosya karari tek bir segmente degil, segmentlerin genel egilimine dayanmaktadir.

`Average Segment Fake Ratio by Class` grafigi bunu destekler:

```text
Real dosyalarda ortalama fake segment orani: yaklasik 0.14
Fake dosyalarda ortalama fake segment orani: yaklasik 0.97
```

Bu, modelin fake dosyalarda segmentlerin neredeyse tamamini fake olarak gordugunu, real dosyalarda ise az sayida segmenti hatali fake olarak isaretledigini gosterir.

## Veri Dengesizligi Neden Tamamen Dengelenmedi?


Veri setini tamamen esit hale getirmek yerine veri dengesizligini egitim stratejisiyle yonetmek tercih edilmistir. Bunun nedeni, cogunluk sinifindan veri silmenin bilgi kaybina neden olabilmesidir. Fake sinifindan cok sayida ornek cikarmak modelin farkli fake uretim bicimlerini ogrenmesini zorlastirabilir.

Bu nedenle veri cesitliligi korunmus, dengesizligin etkisi ise egitimde sinif agirliklari, weighted sampling ve focal loss gibi yontemlerle azaltılmistir. Degerlendirme tarafinda da accuracy tek basina kullanilmamis; precision, recall, F1-score ve confusion matrix ile sinif bazli performans ayrica incelenmistir.

## Precision Tabanli Egitim ve Kullanilan Metrikler

Bu calismada model secimi ve performans yorumu yalnizca accuracy uzerinden yapilmamistir. Bunun temel nedeni veri setindeki real/fake dagiliminin dengeli olmamasidir. Dengesiz veri setlerinde accuracy yuksek gorunebilir; ancak bu durum modelin azinlik sinifini ne kadar iyi yakaladigini her zaman gostermez.

Bu nedenle egitim ve degerlendirme surecinde su metrikler izlenmistir:

```text
Precision
Recall
F1-score
Macro F1
Balanced Accuracy
Bonafide / Real Recall
Spoof / Fake Recall
Confusion Matrix
Segment-level Accuracy
Segment-level Precision
Segment-level Recall
Segment-level F1-score
```

Precision tabanli yaklasimin amaci, modelin fake dedigi orneklerin gercekten fake olma oranini kontrol etmektir. Deepfake tespitinde bu onemlidir; cunku gercek bir sesi yanlislikla fake olarak isaretlemek kullanici guveni ve sistem kullanilabilirligi acisindan ciddi bir problemdir.

Recall da ayrica izlenmistir; cunku fake sesleri kacirmamak da guvenlik acisindan onemlidir. Bu nedenle tek bir metrik yerine precision, recall ve F1-score birlikte yorumlanmistir.


`precision_training_log.csv` dosyasinda egitim boyunca hem dosya seviyesi hem de segment seviyesi metrikler kaydedilmistir. Bu sayede model secimi tek bir sonuca degil, surec boyunca izlenen validasyon performansina dayandirilmistir.

## Threshold Secimi Nasil Aciklanir?

Threshold, modelin fake olasiligini sinif etiketine cevirdigi karar siniridir.

```text
P(fake) >= threshold ise Fake
P(fake) < threshold ise Real
```

Bu calismada threshold test verisine gore elle ayarlanmamistir. En iyi checkpoint kaydedilirken validasyon performansina gore belirlenen threshold kullanilmistir. Bu yaklasim test verisine overfit etmeyi engeller.


Threshold etkisi:

```text
Threshold dusurulurse fake yakalama orani artabilir, ancak real sesleri fake sanma riski yukselir.
Threshold yukseltilirse model daha temkinli fake der, ancak bazi fake ornekler kacabilir.
```

## Scriptler

Performans scripti:

```bash
python VoiceFake_performans.py
```

Bu script ASVspoof dev verisi uzerinde modeli degerlendirir ve `VoiceFake_Performans_Grafikleri*` klasoru altina su dosyalari uretir:

```text
metrics.json
file_predictions.csv
segment_predictions.csv
confusion_file.png
confusion_segment.png
segment_probability_distribution.png
segment_fake_ratio_by_class.png
```


Bu calismada model, dosya seviyesinde cok yuksek performans gostermistir. Segment seviyesinde kismi hatalar gorulse de segmentlerin genel egilimi dosya kararini desteklemektedir. Veri dengesizligi nedeniyle accuracy tek basina yeterli gorulmemis; precision, recall, F1-score, confusion matrix ve segment bazli olasilik dagilimlari birlikte sunulmustur.

Calismanin sinirliligi, real ve fake ornek sayilarinin dengeli olmamasidir. Bu nedenle ileride daha dengeli ve daha genis bir test setiyle ek dogrulama yapilmasi onerilir.

## Veri Seti Secimi?

Bu calismada ASVspoof veri seti tercih edilmistir; cunku ses sahteciligi ve spoofing tespiti alaninda yaygin olarak kullanilan, literaturde kabul goren ve protokol dosyalariyla etiket yapisi net olan bir veri setidir. Bu durum modelin egitim ve degerlendirme surecini daha kontrollu hale getirir.

Farkli veri setlerinin kullanilmamasi tamamen bir eksiklik olarak degil, calisma kapsaminin sinirlandirilmasi olarak aciklanabilir.

Bu calismada tek ve standart bir veri seti uzerinden ilerlemeyi tercih ettim. ASVspoof, bu alanda yaygin kullanilan ve etiket/protokol yapisi net olan bir veri seti oldugu icin kontrollu deney yapmaya uygundu. Farkli veri setleri eklemek calismanin kapsamını genisletirdi; ancak ayni zamanda kayit kosullari ve veri dagilimi farklari nedeniyle sonuclari dogrudan karsilastirmayi zorlastirabilirdi.
