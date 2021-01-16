---
layout: post
title:  "[TR] MLOps Serisi I - Baştan-Sona Makine Öğrenmesi İş Akışının Tanıtılması"
author: "MMA"
comments: true
---

Bir makine öğrenmesi modelini elde etmek çoğu zaman kolay olabilir. Ancak bu modelin gerçek hayatta kullanılabilirliği ve üretimde (prodüksiyonda) son kullanıcıya sunulması zahmetli bir süreçtir. Bu nedenle makine öğrenmesi ile ilgilenenler için MLOps isminde bir seri başlatıyorum. Her hafta makine öğrenmesi operasyonları ile ilgili İngilizce yazılmış çok başarılı bir internet günlüğünü Türkçe'ye çevirip paylaşacağım. 

Serinin ilk çevirisi konuyla ilgili üst düzey bir tanıtım yapan Dr. Larysa Visengeriyeva, Anja Kammer, Isabel Bär, Alexander Kniesz, ve Michael Plöd tarafından yazılmış "An Overview of the End-to-End Machine Learning Workflow" isimli yazı. https://ml-ops.org/content/end-to-end-ml-workflow

MLOps sözcüğü makine öğrenmesi ve operasyon (İng. Operations) sözcüklerinin birleşimidir ve prodüksiyona (üretime, İng. production) sokulan bir Makine Öğrenmesi (veya Derin Öğrenme) modelinin yaşam döngüsünü yönetmeye yardımcı olmak için veri bilimcileri ve operasyon uzmanları arasında iletişim ve işbirliği sağlayan bir uygulamadır. DevOps (Developer Operations - Geliştirici Operasyonları) veta DataOps (Data Operations - Veri Operasyonları)'a çok benzer. Naif bir bakış açısıyla, MLOps sadece makine öğrenimi alanına uygulanan DevOps'tur.

Makine Öğrenmesi operasyonları, makine öğrenmesi modellerinin gelişimini daha güvenilir ve verimli yapmak için gerekli olan tüm süreçleri tanımlayarak makine öğrenmesi modellerinin geliştirilmesine ve dağıtımına (İng. deployment) yardımcı olmak için gerekli ilkelerin belirlenmesi üzerine odaklanır.

# Baştan-Sona Makine Öğrenmesi İş Akışının Tanıtılması

Bu bölümde, makine öğrenmesi tabanlı bir yazılım geliştirmek için gerçekleşmesi gereken tipik bir iş akışını üst düzey bir biçimde gözden geçireceğiz. Genel olarak, bir makine öğrenmesi projesinin hedefi toplanmış veriyi kullanarak ve bu veriye makine öğrenmesi algoritmalarını uygulayarak istatistiksel bir model elde etmektir. Bu nedenle, makine öğrenmesi tabanlı her yazılımın üç ana bileşeni vardır: Veri, bir makine öğrenmesi modeli ve kod. Bu bileşenlere karşılık olarak tipik bir makine öğrenmesi iş akışı üç ana aşamadan oluşmaktadır: 

* **Veri Mühendisliği:** veri toplama & veri hazırlama,
* **Makine Öğrenmesi Model Mühendisliği:** bir makine öğrenmesi modelinin eğitilmesi & servis edilmesi, ve
* **Kod Mühendisliği:** son ürüne elde edilen makine öğrenmesi modelinin entegre edilmesi.

Aşağıdaki şekil tipik bir makine öğrenmesi iş akışında olan temel adımları göstermektedir.
![](ml-engineering.jpg)

## Veri Mühendisliği 

Herhangi bir veri bilimi iş akışındaki ilk adım, analiz edilecek verinin elde edilmesi ve hazırlanmasıdır. Tipik olarak, veri çeşitli kaynaklardan alınır ve bu bu veri farklı formatlara sahip olabilir. Veri hazırlama (İng. data preparation), veri toplama (İng. data acquisition) adımını takip eder ve Gartner'a göre "_veri entegrasyonu, veri bilimi, veri keşfi ve analitik/iş zekası (İng. business intelligence - BI) kullanım senaryoları için  ham veriyi keşfetmek, birleştirmek, temizlemek ve işlenmiş bir veri setine dönüştürmek amacıyla kullanılan yinelemeli ve çevik bir süreçtir._" Veriyi analiz için hazırlama aşaması ara bir aşama olsa da, bu aşamanın kaynaklar ve zaman açısından çok masraflı olduğu bildirilmektedir. Veri hazırlama, veri bilimi iş akışındaki kritik bir işlemdir çünkü veride bulunan hataların bir sonraki aşama olan veri analizine aktarılmasını engellemek önem arz etmektedir. Böyle bir durum veriden yanlış çıkarsamaların elde edilmesiyle sonuçlanacaktır. 

Bir Veri Mühendisliği iletim hattı (İng. Pipeline), makine öğrenmesi algoritmaları için gerekli eğitim ve test kümelerini sağlayacak olan mevcut veri üzerinde yapılacak bir takım operasyonlar dizisini kapsamaktadır:

1. **Veri Alınımı (İng. Data Ingestion)** - Spark, HDFS, CSV, vb. gibi çeşitli programlar ve formatlar kullanarak veri toplama. Bu adım, sentetik veri oluşturmayı veya veri zenginleştirmeyi de içerebilir.
2. **Keşif ve Doğrulama (İng. Exploration and Validation)** - Verilerin içeriği ve yapısı hakkında bilgi almak için veri profili oluşturmayı içerir. Bu adımın çıktısı, maksimum, minimum, ortalama değerler gibi bir meta veri kümesidir. Veri doğrulama operasyonları, bazı hataları tespit etmek için veri setini tarayan, kullanıcı tanımlı hata tespit fonksiyonlarıdır.
3. **Veri Düzeltme (Temizleme) (İng. Data Wrangling (Cleaning))** - Verideki belirli nitelikleri (değişkenleri) yeniden biçimlendirme ve verilerdeki hataları düzeltme süreci (örneğin kayıp değer ataması).
4. **Veri Etiketleme (İng. Data Labeling)** - Her veri noktasının belirli bir kategoriye atandığı Veri Mühendisliği iletim hattının bir operasyonudur.
5. **Veri Ayırma (İng. Data Splitting)** - Verileri, bir makine öğrenmesi modeli elde etmek için temel makine öğrenmesi aşamalarında kullanılacak eğitim, doğrulama ve test veri kümeleri olarak bölme. 

## Model Mühendisliği

Makine öğrenmesi iş akışının temeli, bir makine öğrenmesi modeli elde etmek için makine öğrenmesi algoritmalarını oluşturma ve bu algoritmaları çalıştırma aşamasıdır. Model Mühendisliği iletim hattı, sizi nihai bir modele götüren bir dizi operasyon içerir:

1. **Modelin Eğitilmesi (İng. Model Training)** - Bir makine öğrenmesi modelini eğitmek için bir makine öğrenmesi algoritmasını eğitim verilerine uygulama süreci. Ayrıca, modelin eğitimi için gerekli olan özellik mühendisliği (İng. feature engineering) ve modelin hiperparametrelerini ayarlama adımlarını içerir.
2. **Modelin Değerlendirilmesi (İng. Model Evaluation)** - Bir makine öğrenmesi modelini üretimde (prodüksiyonda) son kullanıcıya sunmadan önce, bu modelin orijinal kodlanmış hedefleri karşıladığından emin olmak için eğitilmiş modelin doğrulanması.
3. **Modelin Testi Edilmesi (İng. Model Testing)** - Eğitim ve doğrulama kümeleri dışında bulunan diğer tüm veri noktalarını kullanarak son "Model Kabul Testi"ni gerçekleştirme.
4. **Modeli Paketleme (İng. Model Packaging**) - İş uygulaması tarafından kullanılsın diye, nihai makine öğrenmesi modelinin belirli bir formata (örneğin PMML, PFA veya ONNX) aktarılması işlemi.

## Model Dağıtımı

Bir makine öğrenmesi modelini eğittikten sonra bu modeli bir mobil veya masaüstü uygulaması gibi bir iş uygulamasının parçası olarak dağıtmamız gerekir. Makine öğrenmesi modelleri, tahminler üretmek için çeşitli veri noktalarına (özellik vektörü) ihtiyaç duyar. Makine öğrenmesi iş akışının son aşaması, önceden tasarlanmış makine öğrenmesi modelinin mevcut yazılıma entegrasyonudur. Bu aşama aşağıdaki operasyonları içerir:

1. **Modelin Servis Edilmesi (İng. Model Serving)** - Üretim (prodüksiyon) ortamında bir makine öğrenmesi modelinin yapısının ele alınması süreci.
2. **Modelin Performansını İzleme (İng. Model Performance Monitoring)** - Bir makine öğrenmesi modelinin performansını, tahmin yaparak veya öneri sunarak canlı (İng. live) ve önceden görülmemiş verilere (İng. previously unseen data) dayalı olarak gözlemleme süreci. Özellikle, önceki modelin performansından tahmin sapması gibi makine öğrenmesine özgü göstergeler ile ilgileniyoruz. Bu göstergeler, modelin yeniden eğitilmesi için bize uyarı niteliğinde olabilir.
3. **Model Performansı Günlüğü (İng. Model Performance Logging)** - Her çıkarım talebi günlük (İng. log) kaydı ile sonuçlanır.

**Bu çevirinin izinsiz ve kaynak gösterilmeden kullanılması yasaktır.**
