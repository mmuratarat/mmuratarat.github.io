---
layout: post
title:  "[TR] MLOps Serisi III - Bir Makine Öğrenmesi Yazılımının Üç Aşaması"
author: "MMA"
comments: true
---

MLOps (Makine Öğrenmesi operasyonları) isimli serinin üçüncü kısımı, ilk yazıda olduğu gibi, Dr. Larysa Visengeriyeva, Anja Kammer, Isabel Bär, Alexander Kniesz, ve Michael Plöd tarafından INNOQ için yazılmış "[Three Levels of ML Software](https://ml-ops.org/content/three-levels-of-ml-software){:target="_blank"}" isimli yazı. İyi Okumalar!

# Bir Makine Öğrenmesi Yazılımının Üç Aşaması

Makine Öğrenmesi / Yapay Zeka, yeni uygulamalar ve endüstriler tarafından hızla benimsenmektedir. Daha önce bahsedildiği gibi, bir makine öğrenmesi projesinin amacı, toplanmış verileri kullanarak ve bu verilere makine öğrenmesi algoritmalarını uygulayarak istatistiksel bir model oluşturmaktır. Ancak, başarılı makine öğrenmesi tabanlı yazılım projeleri oluşturmak halen zordur çünkü her makine öğrenmesi tabanlı yazılımın üç ana bileşeni yönetmesi gerekir: **Veri**, **Model** ve **Kod**. Makine Öğrenmesi Modeli Operasyonelleştirme Yönetimi - **MLOps**, bir DevOps uzantısı olarak, Makine Öğrenmesi modellerini tasarlama, oluşturma ve üretime dağıtma konusunda etkili uygulamalar ve süreçler sunar. Burada, Makine Öğrenmesi tabanlı yazılımın geliştirilmesinde yer alan temel teknik metodolojileri, yani Veri Mühendisliği, Makine Öğrenmesi Model Mühendisliği ve Yazılım Mühendisliği'ni tanımlayacağız. 

Oluşturacağınız iletim hattının her adımında öğrendiğiniz her şeyi **belgelemenizi** öneririz.

## Veri: Veri Mühendisliği İletim Hatları

Herhangi bir makine öğrenmesi iş akışının temel parçasının Veri olduğundan daha önce bahsetmiştik. İyi veri kümelerinin toplanması, Makine Öğrenmesi modelinin kalitesi ve performansı üzerinde büyük bir etkiye sahiptir. Literatürdeki meşhur alıntı

"Çöp içeri çöp dışarı (Garbage In, Garbage Out)",

makine öğrenmesi bağlamında, bir makine öğrenmesi modelinin yalnızca elinizdeki verileriniz kadar iyi olduğu anlamına gelir. Bu nedenle, bir makine öğrenmesi modelinin eğitilmesi için kullanılan veriler dolaylı olarak üretim sisteminin genel performansını etkilemektedir. Veri kümesinin miktarı ve kalitesi genellikle elinizdeki probleme göre değişebilir ve deneysel olarak incelemesi yapılabilir.

Önemli bir adım olan _veri mühendisliği_nin çok zaman alıcı olduğu bildirilmektedir. Bir makine öğrenmesi projesinde zamanımızın çoğunu veri kümeleri oluşturmak, verileri temizlemek ve dönüştürmek için harcayabiliriz. 

Veri mühendisliği iletim hattı, mevcut veriler üzerinde bir işlemler dizisi oluşturmak için yaratılır. Bu işlemlerin nihai amacı, makine öğrenmesi algoritmaları için eğitim ve test veri kümeleri oluşturmaktır. Aşağıda, Veri Alınımı (Data Ingestion), Keşif ve Doğrulama (Exploration and Validation), Veri Düzenleme (Temizleme)(Data Wrangling (Cleaning)) ve Veri Ayırma (Data Splitting) gibi veri mühendisliği iletim hattı oluştururken takip edilmesi gereken her aşamayı açıklayacağız.

### Veri Alınımı (Data Ingestion)

_Veri Alınımı_ - Dahili / harici veritabanları, veri reyonları, OLAP küpleri, veri ambarları, OLTP sistemleri, Spark, HDFS ve bunlar gibi çeşitli sistemleri, çerçeveleri ve formatları kullanarak veri toplama. Bu adım, sentetik veri oluşturmayı veya veri zenginleştirmeyi de içerebilir. Bu adım için en iyi uygulamalar, maksimum düzeyde otomatikleştirilmesi gereken aşağıdaki eylemleri içerir:

<i class="fa fa-arrow-right" aria-hidden="true"></i> Veri Kaynaklarını Tanımlama: Veriyi bulun ve kaynağını belgeleyin<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Alan Tahmini: verinin depolama alanında ne kadar yer kaplayacağını kontrol edin.<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Alan Konumu: Yeterli depolama alanına sahip bir çalışma alanı oluşturun.<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Veri Elde Etme: Verileri alın ve verilerin kendisini değiştirmeden kolayca işlenebilecek bir formata dönüştürün.<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Verileri Yedekleyin: Her zaman verilerin bir kopyası üzerinde çalışın ve orijinal veri kümesini saklayın.<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Gizlilik Uyumluluğu:  Genel Veri Koruma Yasasına (GDPR - General Data Protection Regulation) uyumluluğunu sağlamak için hassas bilgilerin silindiğinden veya korunduğundan (örneğin, anonimleştirildiğinden) emin olun.<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Meta Veri Kataloğu: Boyut, format, diğer adlar (rumuz veya takma ad), son değiştirilme zamanı ve erişim kontrol listeleri gibi temel bilgileri kaydederek veri kümesinin meta verilerini belgelemeye başlayın. (Daha fazla bilgi için [buraya](https://dl.acm.org/doi/pdf/10.1145/2882903.2903730?download=true){:target="_blank"} tıklayınız.)<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Test Verisi: Bir test kümesi örnekleyin, bir kenara koyun ve "_veri gözetleme_ (data snooping)" yanlılığından kaçınmak için ona asla bakmayın. Bu test kümesini kullanarak belirli bir makine öğrenmesi modeli seçiyorsanız hata yapıyorsunuz. Bu, çok iyimser ve üretimde iyi performans göstermeyecek bir makine öğrenmesi modeli seçimine yol açacaktır.

### Keşif ve Doğrulama (Exploration and Validation)

_Keşif ve Doğrulama_ - Verilerin içeriği ve yapısı hakkında bilgi edinmek için veri profili oluşturmayı kapsar. Bu adımın çıktısı, maksimum, minimum, ortalama değerler gibi bir meta veri kümesidir. Veri doğrulama işlemleri, bazı hataları tespit etmek için veri kümesini tarayan kullanıcı tanımlı hata algılama fonksiyonlarıdır. Doğrulama, veri kümesi doğrulama rutinlerini (hata tespit yöntemleri) çalıştırarak verilerin kalitesini değerlendirme sürecidir. Örneğin, "adres" özniteliği (feature) için adres bileşenleri tutarlı mıdır? Adres ile doğru posta kodu ilişkilendirilmiş midir? İlgili niteliklerde (attributes) kayıp değerler var mı? Bu adım için en iyi uygulamalar aşağıdaki eylemleri içerir:

<i class="fa fa-arrow-right" aria-hidden="true"></i> Hızlı Uygulama Geliştirme (RAD - Rapid Application Development) araçlarını kullanın: Jupyter not defterlerini kullanmak, verinin keşfi ve veri üzerinde yapılacak denemelerin kayıtlarını tutmanın iyi bir yoludur. <br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Öznitelik Profili Oluşturma: Her öznitelikle ilgili meta verileri alın ve belgeleyin, örneğin:<br>
> <i class="fa fa-arrow-right" aria-hidden="true"></i> İsim<br>
> <i class="fa fa-arrow-right" aria-hidden="true"></i> Kayıt Sayısı<br>
> <i class="fa fa-arrow-right" aria-hidden="true"></i> Veri Tipi (kategorik, sayısal (nümerik), int (tamsayı) / float (kayan noktalı sayı), metin, yapılandırılmış vb.)<br>
> <i class="fa fa-arrow-right" aria-hidden="true"></i> Sayısal Ölçüler (sayısal veriler için min, maks, ortalama, medyan vb.)<br>
> <i class="fa fa-arrow-right" aria-hidden="true"></i> Kayıp değerlerin miktarı (veya "kayıp değer oranı" = kayıp değerlerin sayısı / Kayıt sayısı)<br>
> <i class="fa fa-arrow-right" aria-hidden="true"></i> Dağılım türü (Gauss, uniform (tekdüze), logaritmik, vb.)<br>

<i class="fa fa-arrow-right" aria-hidden="true"></i> Etiket (Label) Özniteliği Tanımlama: Denetimli öğrenme görevleri için hedef öznitelik(ler)i tanımlayın.<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Veri Görselleştirme: Değer dağılımı için görsel bir temsil oluşturun.<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Nitelik Korelasyonu: Nitelikler arasındaki korelasyonları hesaplayın ve analiz edin.<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Ek Veriler: Modeli oluşturmak için yararlı olacak verileri tanımlayın ("Veri Alınımı" adımına geri dönün).

### Veri Düzenleme (Temizleme)(Data Wrangling (Cleaning))

_Veri Düzenleme (Temizleme)_ - Verilerin şemasının biçimini değiştirebilecek belirli öznitelikleri yeniden biçimlendirerek veya yeniden yapılandırarak, verileri programatik olarak düzenlediğiniz veri hazırlama adımı. Tüm bu fonksiyonellikleri gelecekteki verilerde yeniden kullanmak için veri iletim hattındaki tüm veri dönüşümleri için komut dosyaları (betikler) veya fonksiyonları yazmanızı öneririz.

<i class="fa fa-arrow-right" aria-hidden="true"></i> Dönüşümler: Uygulamak isteyebileceğiniz gelecek vaat eden dönüşümleri tanımlayın.<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Aykırı Değerler: Aykırı değerleri düzeltin veya bu aykırı değerlerden kurtulun (isteğe bağlı).<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Kayıp Değerler: Kayıp değerleri (örneğin, sıfır, ortalama, veya medyan ile) doldurun veya kayıp değer içeren satırları veya sütunları silin.<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Alakasız Veriler: Görev için yararlı bilgiler sağlamayan özniteliklerden kurtulun (öznitelik mühendisliği ile ilgili).<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Verileri Yeniden Yapılandırma: Aşağıdaki işlemleri içerebilir ("[Principles of Data Wrangling](https://learning.oreilly.com/library/view/principles-of-data/9781491938911/){:target="_blank"}" kitabından):<br>

> <i class="fa fa-arrow-right" aria-hidden="true"></i> Sütunları taşıyarak kayıt alanlarını yeniden sıralama<br>
> <i class="fa fa-arrow-right" aria-hidden="true"></i> Değerleri ayıklayarak yeni kayıt alanları oluşturma<br>
> <i class="fa fa-arrow-right" aria-hidden="true"></i> Birden çok kayıt alanını tek bir kayıt alanında birleştirme<br>
> <i class="fa fa-arrow-right" aria-hidden="true"></i> Kayıtların bazılarını silerek veri kümelerini filtreleme<br>
> <i class="fa fa-arrow-right" aria-hidden="true"></i> Veri kümesinin ve kayıtlarla ilişkili alanların ayrıntı düzeyini (granularity), birleştirme (aggregation) ve pivotlama yaparak değiştirme.

### Veri Ayırma (Data Splitting)

Veri Ayırma - Makine Öğrenmesi modelini oluşturmak için temel makine öğrenmesi aşamalarında kullanılacak verileri eğitim (%80), doğrulama ve test kümesi olmak üzere üçe ayırın.

## Model: Makine Öğrenmesi İletim Hattı

Makine öğrenmesi iş akışının temeli, bir makine öğrenmesi modeli elde etmek için makine öğrenmesi algoritmalarını yazma ve çalıştırma aşamasıdır. Model mühendisliği iletim hattı genellikle bir veri bilimi ekibi tarafından kullanılır ve nihai bir modele götüren bir dizi işlemi içerir. Bu işlemler, _Modelin Eğitimi_, _Modelin Değerlendirilmesi_, _Modelin Test Edilmesi_ ve _Modelin Paketlenmesi_ adımlarını içerir. Bu adımları olabildiğince otomatikleştirmenizi öneririz.

### Modelin Eğitimi

_Modelin Eğitimi_ - Bir Makine Öğrenmesi algoritmasını bir modelini eğitmek için eğitim verilerine uygulama süreci. Ayrıca, modelin eğitimi sırasında uygulanması gereken hiperparametrelere ince ayar verilmesi ve öznitelik mühendisliği adımlarını da içerir. Aşağıdaki liste, Aurélien Géron tarafından yazılan "[Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/app02.html#project_checklist_appendix){:target="_blank"}" isimli kitaptan alınmıştır.

<i class="fa fa-arrow-right" aria-hidden="true"></i> Öznitelik mühendisliği şunları içerebilir:

> <i class="fa fa-arrow-right" aria-hidden="true"></i> Sürekli öznitelikleri ayrıklaştırın (kategorikleştirin)<br>
> <i class="fa fa-arrow-right" aria-hidden="true"></i> Öznitelikleri ayrıştırın (örneğin, Kategorik, tarih / saat vb.)<br>
> <i class="fa fa-arrow-right" aria-hidden="true"></i> Özniteliklerin dönüşümlerini ekleyin (örneğin, log (x), sqrt (x), x2, vb.)<br>
> <i class="fa fa-arrow-right" aria-hidden="true"></i> Öznitelikleri birleştirerek kullanışlı yeni özellikler elde edin<br>
> <i class="fa fa-arrow-right" aria-hidden="true"></i> Öznitelik ölçekleme: Öznitelikleri standartlaştırın veya normalleştirin<br>
> <i class="fa fa-arrow-right" aria-hidden="true"></i> Üretime koymak istediğimiz yeni bir özelliğe hızlı bir şekilde geçiş yapmak için yeni öznitelikler hızla eklenmelidir. Daha fazla bilgi için Alice Zheng ve Amanda Casari tarafından yazılan "[Feature Engineering for Machine Learning. Principles and Techniques for Data Scientists](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/){:target="_blank"}" isimli kitaba göz atınız.

<i class="fa fa-arrow-right" aria-hidden="true"></i> Model Mühendisliği, yinelemeli bir süreç olabilir ve aşağıdaki iş akışını içerebilir:

> <i class="fa fa-arrow-right" aria-hidden="true"></i> Her Makine Öğrenmesi modeli spesifikasyonu (bir ML modeli oluşturan kod) bir kod incelemesinden geçmeli ve versiyonlanmalıdır.<br>
> <i class="fa fa-arrow-right" aria-hidden="true"></i> Standart parametreleri kullanarak farklı kategorilerden (örneğin, doğrusal regresyon, lojistik regresyon, k-ortalamalar, naif Bayes, Destek Vektör Makineleri, Rastgele Ağaçlar, vb.) birçok Makine Öğrenmesi modelini eğitin.<br>
> <i class="fa fa-arrow-right" aria-hidden="true"></i> Performanslarını ölçün ve karşılaştırın. Her model için, N-parça çapraz doğrulama kullanın ve N parça üzerinde performans ölçüsünün ortalamasını ve standart sapmasını hesaplayın.<br>
> <i class="fa fa-arrow-right" aria-hidden="true"></i> Hata Analizi: Makine öğrenmesi modellerinin yaptığı hata türlerini analiz edin.<br>
> <i class="fa fa-arrow-right" aria-hidden="true"></i> Daha fazla öznitelik seçimi ve mühendisliği gerçekleştirin.<br>
> <i class="fa fa-arrow-right" aria-hidden="true"></i> Farklı türde hatalar yapan modelleri tercih ederek, en umut vadeden ilk üç ila beş modeli belirleyin.
Çapraz doğrulama kullanarak hiperparametrelerin ayarlanması. Lütfen veri dönüştürme seçeneklerinin de hiperparametreler olduğunu unutmayın. Hiperparametreler için rastgele arama (random search), ızgara aramasına (grid search) tercih edilir.<br>
> <i class="fa fa-arrow-right" aria-hidden="true"></i> _Çoğunluk oylaması_ (majority vote), _torbalama_ (bagging), _hızlandırma_ (boosting) veya _istifleme_ (stacking) gibi Topluluk yöntemlerini (Ensemble methods) göz önünde bulundurun. Makine öğrenmesi modellerini birleştirmek, onları ayrı ayrı çalıştırmaktan daha iyi performans üretmelidir. Daha fazla bilgi için Zhi-Hua Zhou tarafından yazılan "[Ensemble Methods: Foundations and Algorithms](https://www.amazon.com/exec/obidos/ASIN/1439830037/acmorg-20){:target="_blank"}" isimli kitaba göz atınız.

### Modelin Değerlendirilmesi

_Modelin Değerlendirilmesi_ - Makine öğrenmesi modelini üretimde son kullanıcıya sunmadan önce orijinal işletme hedeflerini karşıladığından emin olmak için eğitimli modeli doğrulayın.

### Modelin Test Edilmesi

_Modelin Test Edilmesi_ - Nihai Makine Öğrenmesi modeli eğitildikten sonra, son olarak "Model Kabul Testi" gerçekleştirilerek genelleme hatasını tahmin etmek için daha önceden görülmemiş veri kümesi üzerinde bu modelin performansının ölçülmesi gerekir.

### Modelin Paketlenmesi

_Modelin Paketlenmesi_ - Nihai makine öğrenmesi modelini, herhangi bir uygulama tarafından tüketilecek modeli tanımlayan belirli bir biçime (ör. PMML, PFA veya ONNX) aktarma işlemi. Makine öğrenmesi modelinin nasıl paketlenebileceğini aşağıdaki "Makine Öğrenmesi Modeli serileştirme formatları" bölümünde ele alıyoruz.

#### Makine öğrenmesi iş akışlarının farklı biçimleri

Bir makine öğrenmesi modelinin çalıştırılması, birkaç mimari stil gerektirebilir. Aşağıda, iki boyutta sınıflandırılan dört mimari deseni tartışıyoruz:

1. **Makine Öğrenmesi Modelinin Eğitimi** ve

2. **Makine Öğrenmesi Modelinden Tahmin Yapma**

Konuyu daha basit tutabilmek için, denetimli öğrenme (supervised), denetimsiz öğrenme (unsupervised), yarı denetimli öğrenme (semi-supervised) ve Pekiştirmeli Öğrenme (reinforcement learning) gibi elimizdeki makine öğrenmesi algoritmasının türünü ifade eden üçüncü boyutu yani **3. Makine Öğrenmesi Modelinin Türü**'nü göz ardı ettiğimizi lütfen unutmayın.

**Bir Makine Öğrenmesi Modelinin Eğitimi**ni gerçekleştirmenin iki yolu vardır:

1. Çevrimdışı öğrenme (diğer bir deyişle _yığın_ veya _statik öğrenme_): Model, önceden toplanmış bir dizi veri üzerinde eğitilir. Üretim ortamına dağıtıldıktan sonra, elimizdeki makine öğrenmesi modeli yeniden eğitilene kadar değişmez çünkü model çok sayıda gerçek canlı veri görecek ve kaba tabiri ile _eskiyecektir_. Bu fenomen, "_modelin bozunması_" (model decay) olarak adlandırılır ve model dikkatle takip edilmelidir.
2. Çevrimiçi öğrenme (diğer adıyla _dinamik öğrenme_): Yeni veriler geldikçe model düzenli olarak yeniden eğitilmektedir, örneğin veri akarak yani durmaksızın eş zamanlı geliyorsa. Bu genellikle, makine öğrenmesi modelindeki zamansal etkileri analiz edebilmek için sensör veya hisse senedi alım satım verileri gibi zaman serisi verilerini kullanan makine öğrenmesi sistemleri için geçerlidir.

İkinci boyut, bir makine öğrenmesi modelinden tahminde bulunmak için gerekli mekaniği tanımlayan **Makine Öğrenmesi Modelinden Tahmin Yapma**dır. Bu seçeneği de iki kısıma ayırabiliriz:

1. Yığın tahminler: Üretime dağıtılmış bir makine öğrenmesi modeli, geçmiş girdi verilerine dayalı bir dizi tahmin yapar. Bu genellikle zamana bağlı olmayan veriler için veya çıktı olarak gerçek zamanlı tahminler elde etmenin kritik olmadığı durumlarda yeterlidir.
2. Gerçek zamanlı tahminler (diğer adıyla isteğe bağlı (on-demand) tahminler): Tahminler, talep anında (yani isteğe bağlı olarak) mevcut girdi verileri kullanılarak gerçek zamanlı olarak oluşturulur.

Bu iki boyuta karar verdikten sonra, makine öğrenmesi modellerinin operasyonel hale getirilmesini dört farklı makine öğrenmesi mimarisine sınıflandırabiliriz:

<figure>
  <img src="https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/model_serving_patterns.png?raw=true" alt="my alt text"/>
  <figcaption><small>Bu diagram Dr. Larysa Visengeriyeva tarafından yaratılmış olup, kendisinin izniyle tarafımdan Türkçe'ye çevrilmiştir. İzinsiz kullanılması yasaktır. <a href="https://www.quora.com/How-do-you-take-a-machine-learning-model-to-production" target="_blank">Orijinal şeklin kaynağı</a>.</small></figcaption>
</figure>

Aşağıda Tahmin (Öngörü), İnternet (Web) hizmeti, Çevrimiçi Öğrenme ve AutoML (Otomatikleştirilmiş Makine Öğrenmesi) gibi model mimari türlerinin açıklamasını sunuyoruz.

#### Tahmin (Öngörü)

Bu tür bir makine öğrenmesi iş akışı, akademik araştırmalarda veya veri bilimi eğitimlerinde (örneğin, Kaggle veya DataCamp) geniş çapta kullanılır. Bu form, bir makine öğrenmesi sistemi oluşturmanın en kolay yolu olduğundan bir kaç veri kullanarak makine öğrenmesi algoritmaları ile oynamak için kullanılır. Genellikle, mevcut bir veri kümesini alır, makine öğrenmesi modelini eğitiriz, ardından bu modeli başka (çoğunlukla geçmiş) veriler üzerinde çalıştırırız ve makine öğrenmesi modeli tahminlerde bulunur. Böylelikle bir öngörü elde ederiz. Bu tür makine öğrenmesi iş akışı çok kullanışlı değildir ve bu nedenle üretim sistemleri için (örneğin, mobil uygulamalara dağıtılsın diye) endüstriyel şirketlerde  çok sık kullanılmaz.

#### İnternet (Web) hizmeti

Makine öğrenmesi modellerinin dağıtımı için en yaygın biçimde kullanılan mimari bir internet hizmetidir (mikroservis). Web hizmeti girdi verilerini alır ve bu girdi veri noktaları için bir tahmini geri verir. Model, geçmiş veriler üzerinde çevrimdışı olarak eğitilir, ancak tahminler üretmek için gerçek canlı verileri kullanır. Bir tahminden (yığın tahminler) farkı, bu makine öğrenmesi modelinin neredeyse gerçek zamanlı olarak çalışması ve tüm verileri bir kerede işlemek yerine tek bir kaydı tek seferde işlemesidir. Web hizmeti tahminler yapmak için gerçek zamanlı verileri kullanır, ancak yeniden eğitilene ve üretim ortamına yeniden dağıtılana kadar model sabittir, değişmez.

Aşağıdaki şekil, eğitilmiş modelleri dağıtılabilir hizmetler olarak sarmalamak için kullanılabilecek mimariyi göstermektedir. Dağıtım Stratejileri Bölümünde eğitilmiş makine öğrenmesi modellerini dağıtılabilir hizmetler olarak sarmalamak için kullanılabilecek yöntemlerini tartıştığımızı lütfen unutmayın.

<figure>
  <img src="https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/model-serving-microservices.png?raw=true" alt="my alt text"/>
  <figcaption><small>Burada "GET" bir HTTPS fiilidir. Bu diagram Dr. Larysa Visengeriyeva tarafından yaratılmış olup, kendisinin izniyle tarafımdan Türkçe'ye çevrilmiştir. İzinsiz kullanılması yasaktır.</small></figcaption>
</figure>

#### Çevrimiçi Öğrenme

Bir makine öğrenmesi modelini bir üretim sistemine yerleştirmenin en dinamik yolu, gerçek zamanlı akış analitiği (real-time streaming analytics) olarak da bilinen çevrimiçi öğrenmeyi uygulamaktır. Lütfen çevrimiçi öğrenmenin kafa karıştırıcı bir isim olabileceğini unutmayın çünkü bir makine öğrenmesi modelinin eğitimi genellikle canlı bir sistemde gerçekleştirilmez. Buna _artımlı öğrenme_ (incremental learning) demeliyiz; ancak, _çevrimiçi öğrenme_ terimi Makine Öğrenmesi camiaasında uzun süredir kullanılan oturmuş bir terimdir.  

Bu tür bir makine öğrenmesi iş akışında, bir öğrenme algoritması, veri noktalarını tek tek veya min-yığın olarak adlandırılan küçük gruplar halinde alacak şekilde sürekli bir veri akışına (data stream) sahiptir. Sistem, yeni veriler gelir gelmez anında bu verileri analiz eder, böylece makine öğrenmesi modeli yeni verilerle aşamalı olarak yeniden eğitilir. Bu sürekli olarak yeniden eğitilen model, bir web hizmeti olarak anında kullanılabilir.

Teknik olarak, bu tür bir makine öğrenmesi sistemi, büyük veri sistemlerinde _lambda mimarisiyle_ (lambda architecture) iyi çalışır.

Genellikle, girdi verileri olayların bir akışıdır ve makine öğrenmesi modeli, verileri sisteme girerken alır, bu yeni veriler üzerinde tahminler sağlar ve daha sonra bu yeni verileri de kullanarak öğrenmeyi yeniden başlatır. Model tipik olarak bir Kubernetes kümesi veya benzeri bir sistem üzerinde bir servis olarak çalışır. Üretimdeki bir çevrimiçi öğrenme sistemi ile ilgili büyük bir zorluk, sisteme kalitesiz veriler girerse, makine öğrenmesi modelinin yanı sıra tüm sistem performansının giderek azalacak olmasıdır.

<figure>
  <img src="https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/online_learning.png?raw=true" alt="my alt text"/>
  <figcaption><small>Bu diagram Dr. Larysa Visengeriyeva tarafından yaratılmış olup, kendisinin izniyle tarafımdan Türkçe'ye çevrilmiştir. İzinsiz kullanılması yasaktır. <a href="https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch01.html" target="_blank">Orijinal şeklin kaynağı</a>.</small></figcaption>
</figure>

#### AutoML

Çevrimiçi öğrenmenin daha da karmaşık bir sürümü, otomatikleştirilmiş makine öğrenmesi veya kısaca AutoML'dir. 

AutoML büyük ilgi görmeye başladı ve kurumsal şirketlerde kullanılan Makine Öğremesi için için bir sonraki gelişme olarak kabul ediliyor. AutoML, makine öğrenmesi alanında herhangi bir uzmanlık olmadan minimum çabayla makine öğrenmesi modellerini eğitmeyi vaat ediyor. Kullanıcının sadece veri sağlaması gerekli ve AutoML sistemi, sinir ağı mimarisi gibi bir makine öğrenmesi algoritmasını otomatik olarak seçer ve seçilen algoritmayı yapılandırır.

Modeli güncellemek yerine, üretim ortamında, anında yeni modellerle sonuçlanan eksiksiz bir Makine Öğrenmesi modeli eğitim hattını çalıştırırız. Şimdilik bu, makine öğrenmesi iş akışlarını uygulamanın çok deneysel bir yoludur. AutoML genellikle [Google](https://cloud.google.com/automl/){:target="_blank"} veya [MS Azure](https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml){:target="_blank"} gibi büyük bulut sağlayıcıları tarafından sağlanır. Bununla birlikte, AutoML ile oluşturulan modellerin gerçekten başarılı olması için gereken doğruluk düzeyine ulaşması gerekir.

**Daha fazla okuma için**<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> [AutoML: Genel Bakış ve Araçlar](https://www.automl.org/automl/){:target="_blank"} <br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> [AutoML Karşılaştırması](https://www.researchgate.net/profile/Marc_Andre_Zoeller/publication/332750780_Benchmark_and_Survey_of_Automated_Machine_Learning_Frameworks/links/5e15bd1792851c8364baa47a/Benchmark-and-Survey-of-Automated-Machine-Learning-Frameworks.pdf){:target="_blank"}

#### Bir Makine Öğrenmesi modelini serileştirme formatları

Makine öğrenmesi modellerini dağıtmak için çeşitli formatlar vardır. Dağıtılabilir bir format elde etmek için, Makine Öğrenmesi modelinin mevcut olması ve bağımsız bir nesne olarak çalıştırılabilir olması gerekir. Örneğin, bir Spark işinde Scikit-learn modelini kullanmak isteyebiliriz. Bu, makine öğrenme modellerinin model eğitim ortamının dışında çalışması gerektiği anlamına gelir. Aşağıda, makine öğrenmesi modelleri için, _kullanılan programlama dilinden bağımsız_ ve _Tedarikçiye özgü değişim formatlarınından_ kısaca bahsedeceğiz. 

##### Kullanılan programlama dilinden bağımsız değişim formatları

<i class="fa fa-arrow-right" aria-hidden="true"></i> Birleştirme olarak Türkçe'ye çevrilebilecek olan _Amalgamation_ yöntemi, bir makine öğrenmesi modelini dışa aktarmanın en basit yoludur. Model ve çalıştırılması gereken tüm kodlar tek bir paket olarak birleştirilmiştir. Genellikle, hemen hemen her platformda bağımsız bir program olarak derlenebilen tek bir kaynak kod dosyasıdır. Örneğin, [SKompiler](https://pypi.org/project/SKompiler/){:target="_blank"} kullanarak bir Makine Öğrenmesi modelinin bağımsız bir versiyonunu oluşturabiliriz. Bu python paketi, eğitilmiş Scikit-learn modellerini, SQL sorguları, Excel formülleri, Portable Format for Analytics (PFA) dosyaları veya SymPy ifadeleri gibi diğer formlara  dönüştürmek için bir araç sağlar. Sonuncusu, C, Javascript, Rust, Julia ve bunun gibi çeşitli programlama dillerinde çalıştırılabilecek koda çevrilebilir. Birleştirme (Amalgamation) basit bir kavramdır ve dışa aktarılan makine öğrenmesi modelleri taşınabilirdir. Lojistik regresyon veya karar ağacı gibi bazı kolay makine öğrenmesi algoritmaları için bu biçim kompakttır ve iyi bir performans gösterebilir, bu da kısıtlı gömülü ortamlar için çok kullanışlıdır. Ancak, Bir Makine Öğrenmesi algoritmasına ait model kodun ve bu modelin parametrelerinin birlikte yönetilmesi gerekir.<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> PMML, .pmml dosya uzantısına sahip XML tabanlı bir model servis formatıdır. PMML, [Veri Madenciliği Grubu (DMG)](http://dmg.org/dmg-members.html){:target="_blank"} tarafından standartlaştırılmıştır. Temel olarak .ppml dosya uzantisi, XML'de bir model ve iletim hattını tanımlar. PMML, tüm makine öğrenmesi algoritmalarını desteklemez ve açık kaynak odaklı araçlarda kullanımı lisans sorunları nedeniyle sınırlıdır.<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> PFA (Portable Format for Analytics), PMML'nin yerini alacak şekilde tasarlanmıştır. DMG'den: "_Bir PFA belgesi, skorlama motoru adı verilen bir çalıştırılabilir dosyayı tanımlayan JSON biçimli bir metin dizisidir. Her motorun iyi tanımlanmış bir girdisi, iyi tanımlanmış bir çıktısı ve çıktıyı, ifade merkezli bir sözdizimi ağacında (syntax tree) oluşturmak için girdileri birleştiren fonksiyonları vardır._" (1) , koşul, döngü ve kullanıcı tanımlı fonksiyonlar gibi kontrol yapılarına sahiptir, (2) JSON içerisinde ifade edildiği için, bir PFA formatı, diğer programlar tarafından kolayca oluşturulabilir ve değiştirilebilir, (3) PFA, genişletilebilirlik geri çağırmaları (extensibility callbacks) destekleyen ayrıntılı bir fonksiyon kütüphanesine sahiptir. Makine Öğrenmesi modellerini PFA dosyaları olarak çalıştırmak için PFA'nın etkin olduğu bir ortama ihtiyacımız vardır.<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> ONNX (Open Neural Network eXchange), Makine Öğrenmesi modelinin elde edildiği programdan bağımsız bir dosya formatıdır. ONNX, herhangi bir makine öğrenmesi aracının tek bir model formatını paylaşmasına izin vermek için oluşturulmuştur. Bu format Microsoft, Facebook ve Amazon gibi birçok büyük teknoloji şirketi tarafından desteklenmektedir. Makine öğrenmesi modeli ONNX formatında serileştirildikten sonra, onnx-etkinleştirilmiş çalışma zamanı (runtime) kütüphaneleri (çıkarsama motorları da denir) tarafından tüketilebilir ve ardından tahminlerde bulunabilir. 
[Burada](https://github.com/onnx/tutorials#scoring-onnx-models){:target="_blank"} ONNX formatını kullanabilen araçların listesini bulacaksınız. Özellikle çoğu derin öğrenme aracının ONNX desteğine sahiptir.

[Kaynak: Open Standard Models](https://github.com/adbreind/open-standard-models-2019){:target="_blank"}

##### Tedarikçiye özgü değişim formatları

<i class="fa fa-arrow-right" aria-hidden="true"></i> Scikit-Learn, modelleri .pkl dosya uzantısıyla _pickle_'lenmiş python nesneleri olarak kaydeder.<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> H2O, oluşturduğunuz modelleri POJO (Düz Eski Java Nesnesi - Plain Old Java Object) veya MOJO'ya (Model Nesnesi, Optimize Edilmiş - Model Object, Optimized) dönüştürmenize olanak tanır.<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> MLeap dosya biçiminde kaydedilebilen ve bir MLeap model sunucusu kullanılarak gerçek zamanlı olarak sunulabilen SparkML modelleri. MLeap çalışma zamanı, herhangi bir Java uygulamasında çalışabilen bir JAR'dır. MLeap, eğitime ait iletim hatları ve bu hatları bir MLeap Paketine aktarmak için Spark, Scikit-learn ve Tensorflow'u destekler. TensorFlow, modelleri .pb dosya uzantısıyla kaydeder; ki bu format, protokol tamponu (arabellek) dosyaları uzantısıdır (protocol buffer - protobuf).<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> PyTorch, patentli Torch Script'i kullanarak modelleri bir .pt dosyası olarak servis eder. PyTorch'un model formatı bir C– uygulamasından servis edilebilir.<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Keras, bir modeli .h5 dosyası olarak kaydeder ve bu, bilim camiasında Hierarchical Data Format (Hiyerarşik Veri Biçimi - HDF)'ında kaydedilmiş bir veri dosyası olarak bilinir. Bu dosya türü çok boyutlu veri dizilerini içerir.<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Apple, iOS uygulamalarında gömülü modelleri saklamak için .mlmodel uzantılı patentli dosya biçimine sahiptir. Apple'ın _Core ML_ isimli kütüphanesi, Objective-C ve Swift programlama dillerini desteklemektedir. TensorFlow, Scikit-Learn gibi bir çok diğer makine öğrenmesi kütüphanesinde eğitilen uygulamalara ait makine öğrenmesi model dosyalarını iOS'ta kullanılmak üzere .mlmodel formatına çevirmek için coremltools ve Tensorflow Çeviricisi gibi araçlar kullanmanız gerekmektedir.

Aşağıdaki Tablo, tüm makine öğrenmesi modeli serileştirme formatlarını özetlemektedir:

|                              	| Açık Format 	| Tedarikçi 	| Dosya Uzantısı 	| Lisans 	| MÖ Araçları ve Platformları <br>Desteği 	| İnsan tarafından<br>okunabilir 	| Sıkıştırma 	|
|:----------------------------:	|:-----------:	|:---------:	|:--------------:	|:------:	|:---------------------------------------:	|:------------------------------:	|:----------:	|
| "almagination (birleştirme)" 	| -           	| -         	| -              	| -      	| -                                       	| -                              	|       <i class="fas fa-check" aria-hidden="true"></i>     	|
| PMML                         	|       <i class="fas fa-check" aria-hidden="true"></i>      	| DMG       	| .pmml          	| AGPL   	| R, Python, Spark                        	|                <i class="fas fa-check" aria-hidden="true"></i> (XML)                	|       <i class="fas fa-times" aria-hidden="true"></i>     	|
| PFA                          	|       <i class="fas fa-check" aria-hidden="true"></i>      	| DMG       	| JSON           	|        	| PFA-enabled runtime                     	|            <i class="fas fa-check" aria-hidden="true"></i> (JSON)                    	|      <i class="fas fa-times" aria-hidden="true"></i>      	|
| ONNX                         	|       <i class="fas fa-check" aria-hidden="true"></i>      	| SIG-LFAI  	| .onnx          	|        	| TF, CNTK, Core ML, MXNet, ML.NET        	|               -                 	|      <i class="fas fa-check" aria-hidden="true"></i>      	|
| TF Serving Formatı           	|       <i class="fas fa-check"></i>      	| Google    	| .pf            	|        	| TensorFlow                              	|                <i class="fas fa-check"></i>                	| g-zip      	|
| Pickle Formatı               	|       <i class="fas fa-check"></i>      	|           	| .pkl           	|        	| scikit-learn                            	|                   <i class="fas fa-check"></i>             	| g-zip      	|
| JAR/ POJO                    	|     <i class="fas fa-check"></i>        	|           	| .jar           	|        	| H2O                                     	|                   <i class="fas fa-check"></i>             	|        <i class="fas fa-check"></i>    	|
| HDF                          	|     <i class="fas fa-check"></i>        	|           	| .h5            	|        	| Keras                                   	|                  <i class="fas fa-check"></i>              	|      <i class="fas fa-check"></i>      	|
| MLEAP                        	|      <i class="fas fa-check"></i>       	|           	| .jar/ .zip     	|        	| Spark, TF, scikit-learn                 	|                 <i class="fas fa-check"></i>               	| g-zip      	|
| Torch Script                 	|      <i class="fas fa-times"></i>       	|           	| .pt            	|        	| PyTorch                                 	|                <i class="fas fa-check"></i>                	|     <i class="fas fa-check"></i>       	|
| Apple .mlmodel               	|      <i class="fas fa-times"></i>       	| Apple     	| .mlmodel       	|        	| TensorFlow, scikit-learn, Core ML       	|                  -              	|      <i class="fas fa-check"></i>      	|

<br>
**Daha fazla okuma için**<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> [MÖ Modelleri eğitim dosyası formatları](https://towardsdatascience.com/guide-to-file-formats-for-machine-learning-columnar-training-inferencing-and-the-feature-store-2e0c3d18d4f9/){:target="_blank"} <br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> [Açık Standart Modeller](https://github.com/adbreind/open-standard-models-2019){:target="_blank"}

## Kod: Dağıtım İletim Hatları

Bir Makine öğrenmesi projesini servis etmenin son aşaması aşağıdaki üç adımı içerir:

1. Modelin Servisi - Makine öğrenmesi modelini bir üretim ortamında dağıtma süreci.
2. Modelin Performansını İzleme - Bir makine öğrenmesi modelinin performansını, tahmin veya öneri gibi canlı ve önceden görülmemiş verilere dayalı olarak gözlemleme süreci. Özellikle, önceki model performansından tahmin sapması gibi makine öğrenmesine özgü sinyallerle ilgileniyoruz. Bu sinyaller, modelin yeniden eğitilmesi için uyarıcılar olarak kullanılabilir.
3. Modelin Performans Günlüğü - Her çıkarsama talebi bir günlük kaydı (log) ile sonuçlanır.

Aşağıda, Modeli Servis Etme Kalıplarını ve Model Dağıtım Stratejilerini tartışıyoruz.

### Modeli Servis Etme Kalıpları

Bir üretim ortamında bir makine öğrenmesi modelini servis ederken üç bileşen dikkate alınmalıdır. _Çıkarsama_, _tahminler_i hesaplamak için bir model tarafından alınacak verileri elde etme sürecidir. Bu süreç bir _model_, bu modeli çalıştırmak için bir _yorumlayıcı_ ve _girdi_ verileri gerektirir. Bir makine öğrenmesi sistemini bir üretim ortamına dağıtmak için yapılması gereken iki şey vardır: ilki, otomatik yeniden eğitim ve makine öğrenmesi modelinin dağıtımı için iletim hattını dağıtmak. İkincisi, önceden görülmemiş veriler üzerinde tahmin elde etmek için bir API oluşturmak.

Modelin servis edilmesi, bir makine öğrenmesi modelini bir yazılım sistemine entegre etmenin bir yoludur. Bir makine öğrenmesi modelini üretime sokmak için kullanılabilecek beş tür kalıp arasındaki farkları aşağıda inceliyoruz: **Servis-Olarak-Model** (Model-as-Service), **Bağımlılık-Olarak-Model** (Model-as-Dependency), **Önhesaplamalı** (Precompute), **İsteğe-Bağlı-Model** (Model-on-Demand) ve **Hibrit-Servis** (Hybrid-Serving). Lütfen yukarıda açıklanan model serileştirme formatlarının herhangi bir model servis etme kalıbı için kullanılabileceğini unutmayın.

Aşağıdaki sınıflandırma bu yaklaşımları göstermektedir:


Şimdi, Servis-Olarak-Model, Bağımlılık-Olarak-Model, Önhesaplamalı, İsteğe-Bağlı-Model ve Hibrit-Servis gibi bir makine öğrenmesi modelini üretmeye yönelik kullanılabilecek servis etme kalıplarını göstereceğiz.

#### Servis-Olarak-Model

Servis-Olarak-Model, bir makine öğrenmesi modelini bağımsız bir servis olacak biçimde sarmalayarak gerçekleştirilen yaygın bir kalıptır. Makine öğrenmesi modelini ve yorumlayıcıyı, uygulamaların, bir REST API aracılığıyla istek gönderebileceği veya bir gRPC hizmeti olarak kullanabileceği özel bir web hizmeti içine alabiliriz. Bu kalıp, Tahmin, Web Hizmeti, Çevrimiçi Öğrenme gibi çeşitli makine öğrenmesi iş akışları için kullanılabilir.

<figure>
  <img src="https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/model-as-service.png?raw=true" alt="my alt text"/>
  <figcaption><small>Bu diagram Dr. Larysa Visengeriyeva tarafından yaratılmış olup, kendisinin izniyle tarafımdan Türkçe'ye çevrilmiştir. İzinsiz kullanılması yasaktır. <a href="https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch02.html#project_chapter" target="_blank">Orijinal şeklin kaynağı</a>.</small></figcaption>
</figure>

#### Bağımlılık-Olarak-Model

Bağımlılık-Olarak-Model, bir makine öğrenmesi modelini paketlemenin muhtemelen en basit yoludur. Paketlenmiş bir ML modeli, yazılım uygulaması içinde bir bağımlılık (diğer bir deyişle destek dosyası - dependency) olarak kabul edilir. Örneğin uygulama, bir makine öğrenmesi modelini, tahmin yöntemini çağırıp değerleri geri döndüren geleneksel bir _jar_ dosyası gibi kullanır. Bu tür bir yöntem uygulamasının döndürdüğü değer, önceden eğitilmiş bir makine öğrenmesi modeli tarafından gerçekleştirilen bazı tahminlerdir. Bağımlılık-Olarak-Model yaklaşımı çoğunlukla sadece Tahmin elde etmek için kullanılır.

<figure>
  <img src="https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/model-as-dependency.png?raw=true" alt="my alt text"/>
  <figcaption><small>Bu diagram Dr. Larysa Visengeriyeva tarafından yaratılmış olup, kendisinin izniyle tarafımdan Türkçe'ye çevrilmiştir. İzinsiz kullanılması yasaktır. <a href="https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch02.html#project_chapter" target="_blank">Orijinal şeklin kaynağı</a>.</small></figcaption>
</figure>

#### Önhesaplamalı Servis

Bu tür bir makine öğrenmesi modeli hizmeti, Tahmin MÖ iş akışıyla sıkı bir şekilde ilişkilidir. Önhesaplamalı servis kalıbıyla, önceden eğitilmiş bir makine öğrenmesi modeli kullanır ve gelen veri yığını için tahminleri önceden hesaplarız. Elde edilen tahminler veritabanında saklanır. Bu nedenle, herhangi bir girdi isteği için, tahmin sonucunu almak üzere veritabanını sorgularız.

<figure>
  <img src="https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/precompute-serving-pattern.png?raw=true" alt="my alt text"/>
  <figcaption><small>Bu diagram Dr. Larysa Visengeriyeva tarafından yaratılmış olup, kendisinin izniyle tarafımdan Türkçe'ye çevrilmiştir. İzinsiz kullanılması yasaktır.</small></figcaption>
</figure>

[Daha fazla okuma için: Makine Öğrenmesi Üretime Getirme (Slaytlar)](https://www.slideshare.net/mikiobraun/bringing-ml-to-production-what-is-missing-amld-2020){:target="_blank"}

#### İsteğe-Bağlı-Model

İsteğe-Bağlı-Model kalıbı, bir makine öğrenmesi modelini çalışma zamanında kullanılabilen bir bağımlılık olarak ele alır. Bu makine öğrenmesi modeli, Bağımlılık-Olarak-Model kalıbının aksine, kendi yayınlanma döngüsüne sahiptir ve bağımsız olarak yayınlanır.

Mesaj-aracı (message-broker) mimarisi genellikle bu tür isteğe bağlı model servisi için kullanılır. Mesaj-aracı topoloji mimari kalıbı iki ana mimari bileşen türü içerir: bir _aracı_ (broker) bileşeni ve bir _olay işlemcisi_ (event processor) bileşeni. Aracı bileşeni, olay akışı (event flow) içinde kullanılan olay kanallarını (event channels) içeren merkezi kısımdır. Aracı bileşeninde bulunan olay kanalları mesaj kuyruklarıdır (message queues). Girdi ve çıktı kuyrukları içeren böyle bir mimariyi hayal edebiliriz. Bir mesaj aracısı, bir işlemin bir girdi kuyruğuna _tahmin istekleri_ (prediction-requests) yazmasına izin verir. _Olay işlemcisi_, model servisi çalışma zamanını ve makine öğrenmesi modelini içerir. Bu işlemci aracıya bağlanır, bu istekleri toplu olarak kuyruktan okur ve tahminlerde bulunmak için bunları modele gönderir. Model servisi süreci, tahmin oluşturmayı girdi verileri üzerinde çalıştırır ve sonuçlanan tahminleri çıktı kuyruğuna yazar. Daha sonra, kuyruğa alınmış tahmin sonuçları, tahmin talebini başlatan tahmin servisine gönderilir.

<figure>
  <img src="https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/model-on-demand.png?raw=true" alt="my alt text"/>
  <figcaption><small>Bu diagram Dr. Larysa Visengeriyeva tarafından yaratılmış olup, kendisinin izniyle tarafımdan Türkçe'ye çevrilmiştir. İzinsiz kullanılması yasaktır.</small></figcaption>
</figure>

**Daha fazla okuma için**<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> [Olay odaklı mimari](https://learning.oreilly.com/library/view/software-architecture-patterns/9781491971437/ch02.html){:target="_blank"} <br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> [Gerçek zamanlı makine öğrenmesi uç noktaları için web hizmetleri ve akış karşılaştırması](https://towardsdatascience.com/web-services-vs-streaming-for-real-time-machine-learning-endpoints-c08054e2b18e){:target="_blank"}

#### Hibrit-Servis (Birleştirilmiş Öğrenme)

Hibrit servis olarak da bilinen Birleştirilmiş Öğrenme (Federe Öğrenme de denir - Federated Learning), kullanıcılara bir model servis etmenin başka bir yoludur. Yaptığı şekilde benzersizdir, çıktıyı tahmin eden tek bir model yoktur, aynı zamanda birçok model vardır. Bir sunucuda tutulan modele ek olarak, kullanıcılar kadar çok sayıda model vardır. Sunucudaki _benzersiz_ bir model ile başlayalım. Sunucu tarafındaki model, gerçek dünya verileriyle yalnızca bir kez eğitilir. Her kullanıcı için başlangıç modelini olarak kabul edilir. Ayrıca, nispeten genel olarak eğitilmiş bir modeldir, bu nedenle kullanıcıların çoğu için uygundur. Öte yandan, gerçek özgün modeller olan kullanıcı tarafı modeller var. Mobil cihazlardaki arta donanım standartları nedeniyle cihazların kendi modellerini eğitmesi mümkündür. Yani, cihazlar kendi kullanıcıları için son derece özelleştirilmiş modellerini eğiteceklerdir. Arada bir, cihazlar önceden eğitilmiş model verilerini (kişisel verileri değil) sunucuya gönderir. _Sunucuda bulunan model_ ayarlanacak, böylece tüm kullanıcı topluluğunun gerçek eğilimleri model tarafından ele alınacaktır. Tüm cihazlardan gelen model verileriyle bir daha ayarlanan sunucudaki model, tüm cihazların kullandığı yeni başlangıç modeli olacaktır. Kullanıcıların herhangi bir problem yaşamaması için, sunucu modelinin güncellenmesi, cihaz boştayken, WiFi'ye bağlıyken ve şarj olurken gerçekleşir. Ayrıca testler cihazlar üzerinde yapılır, bu nedenle sunucudan yeni benimsenen model cihazlara gönderilir ve bu modelin işlevselliği test edilir.

Bu öğrenme türünün en büyük yararı, son derece kişisel olan eğitim ve test için kullanılan verilerin, kullanıcı hakkındaki mevcut tüm bilgiyi yakalarken hiçbir zaman cihazlardan dışarı çıkmamasıdır. Bu şekilde, bulutta tonlarca (muhtemelen kişisel) veri depolamak zorunda kalmadan yüksek doğrulukta modeller eğitmek mümkündür. Ancak bedava öğle yemeği diye bir şey yoktur, normal makine öğrenmesi algoritmaları, her zaman eğitim için mevcut olan güçlü donanım üzerinde homojen ve büyük veri kümeleriyle oluşturulur. Federe Öğrenme ile başka koşullar da vardır, mobil cihazlar daha az güçlüdür, eğitim verileri milyonlarca cihaza dağıtılır ve bu cihazlar her zaman eğitim için müsait olmayabilir. Tam olarak bunun için TensorFlow Federated ([TFF](https://medium.com/tensorflow/introducing-tensorflow-federated-a4147aa20041){:target="_blank"}) oluşturulmuştur. TFF, Birleştirilmiş Öğrenme için oluşturulmuş hafif bir TensorFlow türüdür.

<figure>
  <img src="https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/federated-learning.png?raw=true" alt="my alt text"/>
  <figcaption><small>Bu diagram Dr. Larysa Visengeriyeva tarafından yaratılmış olup, kendisinin izniyle tarafımdan Türkçe'ye çevrilmiştir. İzinsiz kullanılması yasaktır. <a href="https://ai.googleblog.com/2017/04/federated-learning-collaborative.html" target="_blank">Orijinal şeklin kaynağı</a>.</small></figcaption>
</figure>

### Dağıtım Stratejileri

Aşağıda, eğitilmiş modelleri dağıtılabilir hizmetler olarak sarmalamaya yönelik yaygın yöntemleri, yani makine öğrenmesi modellerini Docker Konteynerleri olarak _Bulutta bulunan Sunucularına_ (cloud instances) ve _Sunucusuz Fonksiyonlar_ (serverless function) olarak dağıtmanın yaygın yollarını tartışıyoruz.

#### Makine Öğrenmesi Modellerini Docker Konteyner olarak Dağıtma

Günümüzde bir makine öğrenmesi modelinin dağıtımına yönelik standart, açık bir çözüm yoktur. Ancak, makine öğrenmesi modelinden yapılacak çıkarsama, herhangi bir durum ifadesi taşımadığından ve hafif ve idempotent olarak kabul edildiğinden, konteynerleştirme, ürünün teslimatı için fiili standart haline gelmiştir. Bu, bir makine öğrenmesi modelinin çıkarsama kodunu sarmalayan bir konteyner dağıtmamız gerektiği anlamına gelir. Şirket içi (on-premise), bulut veya hibrit dağıtımlar için Docker, fiili standart konteynerleştirme teknolojisi olarak kabul edilir.

Her zaman gerçekleştirebileceğiniz bir yol, tüm makine öğrenmesi teknoloji yığınını (destek dosyaları dahil) ve makine öğrenmesi modelinden tahmin yapan kodu bir Docker konteynerinde paketlemektir. Ardından, Kubernetes veya bir başka alternatifi (ör. AWS Fargate) gerekli düzenlemeleri (orchestration) gerçekleştirir. Tahmin elde etme gibi bir makine öğrenmesi modelinden elde edilecek fonksiyonellik, daha sonra bir REST API aracılığıyla kullanılabilir (örneğin, [Flask uygulaması](https://flask.palletsprojects.com/en/1.1.x/){:target="_blank"} olarak gerçekleştirilebilir).

<figure>
  <img src="https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/infra-cloud.png?raw=true" alt="my alt text"/>
  <figcaption><small>Bu diagram Dr. Larysa Visengeriyeva tarafından yaratılmış olup, kendisinin izniyle tarafımdan Türkçe'ye çevrilmiştir. İzinsiz kullanılması yasaktır.</small></figcaption>
</figure>

#### Makine Öğrenmesi Modellerini Sunucusuz Fonksiyonlar Olarak Dağıtma

Çeşitli bulut tedarikçileri halihazırda makine öğrenmesi platformları sağlamaktadır. Böylelikle modelinizi servisleriyle birlikte kolayca dağıtabilirsiniz. Amazon AWS Sagemaker, Google Cloud AI Platformu, Azure Machine Learning Studio ve IBM Watson Machine Learning verilebilecek bazı örneklerdir. Ticari bulut hizmetleri, AWS Lambda ve Google App Engine servlet host gibi servisler kullanarak ML modellerinin konteynerleştirmesini de sağlar.

Bir makine öğrenmesi modelini sunucusuz bir fonksiyon (serverless function) olarak dağıtmak için, uygulama kodu ve destek dosyaları tek bir giriş noktası fonksiyonu ile .zip dosyaları halinde paketlenir. Bu fonksiyon daha sonra Azure Functions, AWS Lambda veya Google Cloud Functions gibi büyük bulut sağlayıcıları tarafından yönetilebilir. Ancak, nesnenin büyüklüğü gibi dağıtılan nesnelerin olası kısıtlamalarına dikkat edilmelidir.

<figure>
  <img src="https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/infra-lambda.png?raw=true" alt="my alt text"/>
  <figcaption><small>Bu diagram Dr. Larysa Visengeriyeva tarafından yaratılmış olup, kendisinin izniyle tarafımdan Türkçe'ye çevrilmiştir. İzinsiz kullanılması yasaktır.</small></figcaption>
</figure>

**Bu çevirinin ve çevirideki grafiklerin izinsiz ve kaynak gösterilmeden kullanılması yasaktır.**
