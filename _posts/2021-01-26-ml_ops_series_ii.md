---
layout: post
title:  "[TR] MLOps Serisi II - Burada çözmeye çalıştığımız iş sorunu nedir?"
author: "MMA"
comments: true
---

MLOps (Makine Öğrenmesi operasyonları) isimli serinin ikinci kısımı, ilk yazıda olduğu gibi, Dr. Larysa Visengeriyeva, Anja Kammer, Isabel Bär, Alexander Kniesz, ve Michael Plöd tarafından INNOQ için yazılmış "[What is the business problem that we are trying to solve here?](https://ml-ops.org/content/phase-zero){:target="_blank"}" isimli yazı. İyi Okumalar!

# "Burada çözmeye çalıştığımız iş sorunu nedir?"

Herhangi bir yazılım projesinde en önemli aşama, iş problemini (İng. Business Problem) anlamak ve gereksinimleri oluşturmaktır. Makine Öğrenmesi tabanlı bir yazılım geliştirirken de bu durum çok da farklı değildir. İlk adım iş sorunlarının ve gereksinimlerinin kapsamlı bir incelemesini içerir. Bu gereksinimler, model hedeflerine ve model çıktılarına dönüştürülür. Modelin yayınlanması ve piyasaya sürülmesi için olası hatalar ve minimum başarı belirlenmelidir. Yapay Zeka / Makine Öğrenmesi çözümü üzerinde çalışmaya devam etmek için sorulması gereken en yararlı soru "***Yanlış tahminler ne kadar maliyetlidir?***" Bu soruyu yanıtlamak, bir Makine Öğrenmesi projesinin fizibilitesini tanımlayacaktır.

## İş Akışı'nın Parçalanması

Yapılacak tahminin (Makine Öğrenmesi modelinin) nerede uygulanabileceğini görmek için tüm iş sürecinin her bir görevinin kurucu unsurlarına parçalanması gerekir

| ![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/ml_workflow_decomposition.png?raw=true) | 
|:--:| 
| Bu diagram Dr. Larysa Visengeriyeva tarafından yaratılmış olup, kendisinin izniyle tarafımdan Türkçe'ye çevrilmiştir. İzinsiz kullanılması yasaktır.|


"Yapay Zeka / Makine Öğrenmesi nasıl uygulanır" sorusunu yanıtlamak için aşağıdaki adımları izleriz:

1. Yapay Zeka / Makine Öğrenmesi tarafından güçlendirilebilecek somut **süreci** tanımlayın (yukarıdaki Şekil'e bakınız).
2. Bu süreci yönlendirilmiş bir **görevler** grafiğine parçalayın.
3. İnsan etkisinin görevin neresinden çıkarılabileceğini belirleyin, yani, Makine Öğrenmesi modeli gibi bir tahmin öğesi hangi görevin yerini alabilir?
4. Her bir görevi gerçekleştirmek üzere bir Yapay Zeka / Makine Öğrenmesi aracı uygulamak için Yatırım Karını (İng. ROI - Return On Investment) hesaplayın
5. Her **görev** için ayrı ayrı hesaplanan yatırım karlarına göre Yapay Zeka / Makine Öğrenmesi uygulamalarını sıralayın.
6. Listenin en başından başlayın ve her bir görev için Yapay Zeka Şablonunu veya Makine Öğrenmesi Şablonunu tamamlayarak Yapay Zeka / Makine Öğrenmesi uygulamasını yapılandırın.

Yapay Zeka Şablonu veya alternatifi Makine Öğrenimi Şablonu, parçalanma sürecini yapılandırmaya destek ve yardımcı olur. Ayrıca, tahmin yapmak için tam olarak neyin gerekli olduğunu ve tahmin algoritması tarafından yapılan hatalara nasıl tepki vereceğimizi ifade etmeye yardımcı olurlar.

## Yapay Zeka Şablonu

Yapay Zeka Şablonu, A. Agrawal ve arkadaşları tarafından 2018 yılında yayınlanan “Prediction Machines. The Simple Economics of Artificial Intelligence.” isimli kitapta önerilmiştir ve "Yapay Zeka araçlarını düşünmek, oluşturmak ve değerlendirmek için bir destek görevi görür." Bu tür bir şablon örneği ve her bileşeninin açıklaması aşağıdaki Şekilde verilmiştir:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/AI-Canvas.jpg?raw=true)

[Şekil Kaynağı](https://hbr.org/2018/04/a-simple-tool-to-start-making-decisions-with-the-help-of-ai){:target="_blank"}

## Makine Öğrenmesi Şablonu

Yukarıdaki Yapay Zeka Şablonu, bir Yapay Zeka / Makine Öğrenmesi uygulamasının üst düzey bir yapısını temsil etse de, bir noktada hem makine öğrenmesi sistemi vizyonunu hem de sistemin özelliklerini belirlemek isteriz. Bu hedeflere ulaşmak için kullanılabilecek, [Louis Dorard](https://www.louisdorard.com/){:target="_blank"} tarafından önerilen Makine Öğrenmesi Şablonu adlı başka bir araç daha var. Bu şablon, bir makine öğrenmesi projesini yapılandırır ve projenin gerçekleştirilmesi için gerekli temel gereksinimleri belirlemeye yardımcı olur.
Başlangıçta, "tahmini sistemin son kullanıcıları için neyi başarmak istiyoruz?" Sorusunu yanıtlayarak amacımızı belirleriz. Ardından, iş hedefini makine öğrenmesi görevine ilişkilendiririz.

Şablonun merkezi kısmı, müşteriler için bir miktar değer yaratan ürün veya hizmetleri tanımlayan _Değer Önerisi_ yapı taşıdır. Tipik olarak şu soruları yanıtlarız: _Hangi_ sorunları çözmeye çalışıyoruz? Gerçekleştireceğimiz görev _neden_ önemlidir? Sistemimizin son kullanıcısı _kimdir_? Bu Makine Öğrenmesi projesi son kullanıcıya hangi değeri sağlar? Çıktılarınızı / tahminlerinizi bu kullanıcılar nasıl kullanacaklar?

Şablonun gerı kalanı üç geniş kategoriye ayrılmıştır: _Öğrenme_, _Tahmin_ ve _Değerlendirme_. Öğrenme kategorisi, Makine Öğrenmesi modelinin nasıl öğrenileceğini belirlemekten sorumludur. Tahmin bölümü tahminin nasıl yapıldığını açıklar. Son olarak, Değerlendirme kategorisi, makine öğrenimi modelinde ve sistem değerlendirmesinde kullanılacak yöntemleri ve metrikleri içerir. Aşağıdaki Makine Öğrenmesi Şablonu, Louis Dorard tarafından sağlanan bir örnektir:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/ML_canvas_v04.jpg?raw=true)

Yukarıda bahsedilen Makine Öğrenmesi Şablonu toplamda, _Değer Önerisi_, _Veri Kaynakları_, _Tahmin Görevi_, _Öznitelikler (Mühendisliği)_, _Çevrimdışı Değerlendirme_, _Kararlar_, _Tahminlerde Bulunma_, _Veri Toplama_, _Model Oluşturma_ ve _Anlık Değerlendirme ve İzleme_ gibi on bileşik blok olarak yapılandırılmıştır. Bu blokların her biri, ileride gerçekleştirilecek olan bir makine öğrenmesi uygulamasının bir yönüne odaklanmıştır:

## Değer Önerisi

Bu, şablonun tamamındaki en önemli blokdur. Burada üç önemli soruyu cevaplamalıyız:

1. Sorun _nedir_? Hangi amaca hizmet ediyoruz? Son kullanıcı için ne yapmaya çalışıyoruz?
2. Bu problem _neden_ önemlidir?
3. Son kullanıcı _kimdir_? Bu kullanıcıyı tanımlayabilir miyiz?

To create an effective Value Proposition statement, we could use the Geoffrey Moore’s value positioning statement template:

Etkili bir Değer Önerisi ifadesi oluşturmak için [Geoffrey Moore’un değer konumlandırma ifadesi şablonunu](https://the.gt/geoffrey-moore-positioning-statement/){:target="_blank"} kullanabiliriz:

\** **(İhtiyaç veya fırsat) sahibi olan (hedef müşteri) için bizim (ürün / hizmet adı), (fayda) sağlayan (ürün kategorisi)'dir.** \** 

Çözüm bulmaya çalıştığımız problemin etki alanını daraltmak, ihtiyacımız olan verilerle ilgili bir sonraki soru için faydalı olabilir. Örneğin, genel bir sohbet botu (robotu) oluşturmak yerine, konferans görüşmeleri planlamaya yardımcı olan bir bot oluşturun.

## Veri Kaynakları

Veri, Makine Öğrenmesi modellerini eğitmek için gereklidir. Bu blokta, bir makine öğrenmesi görevi için kullanılabilecek tüm mevcut ve olası veri kaynaklarına açıklık getiriz. Örnek olarak aşağıdaki kaynakları kullanmayı düşünebiliriz:

<i class="fa fa-arrow-right" aria-hidden="true"></i> Dahili/Harici veritabanları.,<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Veri reyonları, OLAP küpleri (İng. Data marts), veri ambarları, OLTP sistemleri.<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Hadoop kümeleri,<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Veri toplamak için kullanılabilecek REST API'leri.<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Statik dosyalar, elektronik tablolar.<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Web kazıma.<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Diğer (Makine Öğrenmesi) sistemlerin çıktısı.<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Açık kaynak veri kümeleri.<br>
&nbsp;<i class="fa fa-arrow-right" aria-hidden="true"></i> Herkese açık faydalı veri kümeleri: [https://www.kaggle.com/datasets](Kaggle Veri Kümeleri), [https://datasetsearch.research.google.com/](Google'ın Veri Kümesi Araması), [https://archive.ics.uci.edu/ml/datasets.php](UCI Veri Havuzu) veya [https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research](makine öğrenmesi araştırması için Wikipedia'nın veri kümeleri listesi).


Bunlara ek olarak, bir makine öğrenmesi uygulamasının gizli maliyetlerine de bakmamız gerekir.

<i class="fa fa-arrow-right" aria-hidden="true"></i> Bu verileri depolamak ne kadar masraflı olacaktır?<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Dış (harici) veri satın almalı mıyız?<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Veriyi diğer sistemlerden erişilebilir kılmak için hangi veri değerlendirme araçları ve süreçleri mevcuttur?

## Tahmin Görevi 

Elimizde hangi verilerin mevcut olduğunu netleştirdikten sonra, ne tür makine öğrenmesi kullanılması gerektiği üzerine beyin fırtınası yapıyoruz. Yapılacak makine öğrenmesi görevini açıklığa kavuşturabilecek soruların bazı örnekleri aşağıda verilmiştir:

<i class="fa fa-arrow-right" aria-hidden="true"></i> Denetimli (eğitimli) veya Denetimsiz (eğitimsiz) öğrenme?<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Bu bir anomali tespiti mi?<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Sorun hangi seçeneğin seçilmesi gerektiği ile mi ilgili? (öneri)<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Sürekli bir değer tahmin etmemiz gerekiyor mu? (regresyon)<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Hangi kategorinin tahmin edilmesi gerekiyor? (sınıflandırma)<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Verilerimizi gruplamamız gerekiyor mu? (kümeleme)<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Denetimli öğrenme kullanılacaksa, ne tür bir makine öğrenmesi görevi seçilmelidir: sınıflandırma, regresyon veya sıralama (İng. ranking)?<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Sınıflandırma yapılacak ise, ikili veya çok sınıflı sınıflandırma görevi mi olacak?<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Bir tahmin görevinin girdisi nedir?<br>
&nbsp;&nbsp;<i class="fa fa-arrow-right" aria-hidden="true"></i> örneğin, E-posta metini.<br>

<i class="fa fa-arrow-right" aria-hidden="true"></i> Tahmin görevinin çıktısı nedir?<br>
&nbsp;&nbsp;<i class="fa fa-arrow-right" aria-hidden="true"></i> örneğin, "İstenmeyen e-posta" ve "normal"<br>

<i class="fa fa-arrow-right" aria-hidden="true"></i> Makine Öğrenmesi modelimizin alabileceği karmaşıklık derecesi (İng. degree of complexity) nedir?
örneğin. modelimiz diğer makine öğrenimi modellerinin bir kombinasyonu mu? Topluluk öğrenmesi (İng. Ensemble Learning) kullanıyor muyuz? Derin öğrenme modelinde kaç tane gizli katman var?<br>

<i class="fa fa-arrow-right" aria-hidden="true"></i> Yukarıdaki modeller için eğitim ve çıkarım süresi gibi karmaşıklık maliyetleri nelerdir?

## Öznitelikler (Mühendisliği)

Her Makine Öğrenmesi algoritmasına verilen verinin satırlarda gözlemler sütunlarda değişkenler olacak şekilde bir formda olması gerektiğinden, girdi verilerinin nasıl temsil edilmesi gerektiğini netleştirmeliyiz.

<i class="fa fa-arrow-right" aria-hidden="true"></i> Ham (işlenmemiş) kaynaklardan öznitelikleri (değişkenleri) nasıl çıkarırız?<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Belirli bir Makine Öğrenmesi görevinde, verinin hangi yönlerinin önemli olduğunu belirlemek için alan uzmanlarını dahil etmeyi düşünün.

## Çevrimdışı Değerlendirme

Bir Makine Öğrenmesi modelinin eğitiminin herhangi bir uygulamasından önce, modelin dağıtımı (İng. deployment) önce sistemi değerlendirmek için eldeki metodları ve metrikleri belirlememiz ve ayarlamamız gerekir. Burada şunları belirlememiz gerekir:

<i class="fa fa-arrow-right" aria-hidden="true"></i> Makine öğrenimi modelinin dağıtımını doğrulayan, alana özgü metrikler. Örneğin, eğitim ve test verileriyle simüle edildiğinde, modelin tahmini, "geleneksel" yolla elde edilen hasılattan daha fazla hasılat sağlar.<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Hangi teknik değerlendirme ölçütleri kullanılmalıdır?<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<i class="fa fa-arrow-right" aria-hidden="true"></i> Kesinlik (İng. Precision), Duyarlılık (İng. Recall/Sensitivity), F-1 ölçütü.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<i class="fa fa-arrow-right" aria-hidden="true"></i> Doğruluk oranı.

<i class="fa fa-arrow-right" aria-hidden="true"></i> Yalancı pozitifler (İng. false positives) ve yalancı negatifler (İng. false negatives) gibi model tahmin hatalarının anlamı nedir?<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Test verimiz nedir?<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Makine öğrenmesi modelinin iyi performans gösterdiğinden emin olmak için ne kadar büyüklükte test verisine ihtiyacımız var?

## Kararlar 

Makine Öğrenmesi görevini, Öznitelik mühendisliğini ve değerlendirme detaylarını tamamladıktan sonra, bir sonraki adım şunları belirlemektir:

<i class="fa fa-arrow-right" aria-hidden="true"></i> Karar vermek için tahminler nasıl kullanılır?<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Son kullanıcı veya sistem, model tahminleriyle nasıl etkileşim kurar?<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<i class="fa fa-arrow-right" aria-hidden="true"></i> Örneğin. Kullanıcı bir ürün önerileri listesi alırsa ne olur? Gelen e-posta "istenmeyen posta" olarak sınıflandırılırsa ne olur?

<i class="fa fa-arrow-right" aria-hidden="true"></i> Karar vermede işleminde herhangi bir gizli maliyet var mıdır? örneğin işin içinde olan bir insan faktörü.

Bu tür bilgiler daha sonra bu Makine Öğrenmesi modelinin nasıl dağıtılacağına karar vermek için gereklidir

## Tahminlerde Bulunma

Bu blok, yeni girdiler üzerinde ne zaman bir tahmin yapmamız gerektiğine ilişkin bilgileri içerir.

<i class="fa fa-arrow-right" aria-hidden="true"></i> Tahminler ne zaman elde edilmelidir?<br>
&nbsp;&nbsp;<i class="fa fa-arrow-right" aria-hidden="true"></i> Kullanıcı uygulamayı her açtığı zaman yeni tahminler yapılır, örneğin  bir ürün önerileri listesi.<br>
&nbsp;&nbsp;<i class="fa fa-arrow-right" aria-hidden="true"></i> Talep üzerine yeni tahminler yapılır.<br>
&nbsp;&nbsp;<i class="fa fa-arrow-right" aria-hidden="true"></i> Bir zamanlama çizelgesine göre yeni tahminler yapılır.

<i class="fa fa-arrow-right" aria-hidden="true"></i> Tahminler her veri noktası için mi yoksa girdi verilerinin bir yığını için mi _anında_ yapılıyor mu?<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Uygulamada, _model çıkarsaması_ hesaplama açısından ne kadar karmaşıklaşır?<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> İşin içinde tahminlerde bulunmayı destekleyen bir insan var mı?

## Veri Toplama

Tahmin Yapma ile ilişkili olarak Veri Toplama bloğu, Makine Öğrenmesi modelini yeniden eğitmek için toplanması gereken yeni veriler hakkında bilgi toplar. Böylelikle, bu makine öğrenmesi modelinin bozulmasını nasıl önlediğimizi belirliyoruz. Bu blokta cevaplanacak diğer sorular şunlardır:

<i class="fa fa-arrow-right" aria-hidden="true"></i> Yeni veriyi nasıl etiketleriz?<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Yeni veri toplamak ne kadar masraflıdır?<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Görüntü, ses veya video gibi zengin medya formatlarını işlemek ne kadar masraflıdır?<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Gelen verilerin manuel olarak temizlenmesi ve etiketlenmesi için kullanılan bir insan faktörü var mı?

## Model Oluşturma

Bir önceki blokla sıkı bir şekilde ilişkili olan Model Oluşturmak bloğu, Makine Öğrenmesi modellerini güncellemeyle ilgili soruları yanıtlar çünkü farklı Makine Öğrenmesi görevleri için bir modelin yeniden eğitilmesi farklı sıklıklarla gerçekleşebilir:

<i class="fa fa-arrow-right" aria-hidden="true"></i> Model ne sıklıkla yeniden eğitilmelidir?<br>
&nbsp;&nbsp;<i class="fa fa-arrow-right" aria-hidden="true"></i> Örneğin, saatlik, haftalık veya yeni bir veri noktası geldiği her an.<br>

<i class="fa fa-arrow-right" aria-hidden="true"></i> Modelin yeniden eğitiminin _gizli maliyetleri_ nelerdir?<br>
&nbsp;&nbsp;<i class="fa fa-arrow-right" aria-hidden="true"></i> örneğin Bu tür görevleri gerçekleştirmek için bulut kaynaklarını kullanıyor muyuz?<br>
&nbsp;&nbsp;<i class="fa fa-arrow-right" aria-hidden="true"></i> bulut hizmetini sağlayanın şirketin fiyat politikası nedir?<br>
&nbsp;&nbsp;<i class="fa fa-arrow-right" aria-hidden="true"></i> donanımsal maliyetlerin tahminini nasıl yapmalıyız?<br>
&nbsp;&nbsp;<i class="fa fa-arrow-right" aria-hidden="true"></i> yaygın Bulut Fiyatlandırma Hesaplayıcıları; [Google Cloud Calculator](https://cloud.google.com/products/calculator), [Amazon ML Pricing](https://docs.aws.amazon.com/machine-learning/latest/dg/pricing.html), [Microsoft Azure Calculator](https://azure.microsoft.com/en-in/pricing/calculator/)'dir

<i class="fa fa-arrow-right" aria-hidden="true"></i> Modeli yeniden eğitmek ne kadar sürer?<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Daha karmaşık ve maliyetli olabileceğinden, bulut operasyonlarının ölçeklendirme sorunlarıyla nasıl başa çıkacağız?<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Teknoloji yığınında değişiklik yapmayı planlıyor muyuz?<br>
&nbsp;&nbsp;<i class="fa fa-arrow-right" aria-hidden="true"></i> örneğin, Modern Yapay Zekada yeni araçlar ve geliştirme iş akışları ortaya çıktıkça teknoloji yığını evrimiyle nasıl başa çıkabiliriz?

## Anlık Değerlendirme ve İzleme

Dağıtımdan sonra, bir makine öğrenmesi modeli değerlendirilmeli ve burada birbiriyle ilişkili olması gereken hem model hem de iş ölçütlerini (metrikler) belirlememiz gerekir. Genel olarak, bu ölçütler S.M.A.R.T metodolojisini takip etmeli ve: Spesifik (İng. Specific), Ölçülebilir (İng. Measurable), Ulaşılabilir (İng. Achievable), İlgili (İng. Relevant) ve Zamana bağlı (İng. Time-bound) olmalıdır.

<i class="fa fa-arrow-right" aria-hidden="true"></i> Sistemin performansını nasıl takip ederiz?<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<i class="fa fa-arrow-right" aria-hidden="true"></i> örneğin, A/B testi<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Değer yaratmayı (İng. value creation) nasıl değerlendiririz?<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<i class="fa fa-arrow-right" aria-hidden="true"></i> örneğin, kullanıcılar gelen kutusunda daha az zaman harcadı.



Bu aşamadaki çıktı, tamamlanmış Makine Öğrenmesi Şablonu'dur. Bu şablonu doldurma çabası, makine öğrenmesi tabanlı yazılımının gerçek amacına ve gizli maliyetlerine ilişkin varoluşsal bir tartışma başlatabilir. Böyle bir tartışma, Yapay Zeka'yı / Makine Öğrenmesi'ni hiç uygulamama kararıyla sonuçlanabilir. Olası nedenler aşağıdakiler gibi olabilir:

<i class="fa fa-arrow-right" aria-hidden="true"></i> Sorunumuzun çözümü yanlış tahminlere müsamaha göstermez.<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Yapay Zeka'yı / Makine Öğrenmesi'ni uygulamak düşük yatırım karı (İng. ROI - Return On Investment) yaratacaktır.<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> Yapay Zeka / Makine Öğrenmesi projesinin bakımı garanti edilmez


Sorulması gereken başka bir soru da gerçekleştirilen Yapay Zeka / Makine Öğrenmesi uygulamasının ne zaman dağıtılması gerektiğidir. Aşağıdaki Şekil, bir Makine Öğrenmesi modelinin erken ve geç dağıtımının dengesini göstermektedir.

| ![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/When-to-deploy-ai.png?raw=true) | 
|:--:| 
| Bu diagram Dr. Larysa Visengeriyeva tarafından yaratılmış olup, kendisinin izniyle tarafımdan Türkçe'ye çevrilmiştir. İzinsiz kullanılması yasaktır.|

<br><br>
Daha fazla okuma

<i class="fa fa-arrow-right" aria-hidden="true"></i> ["What is THE main reason most ML projects fail?"](https://towardsdatascience.com/what-is-the-main-reason-most-ml-projects-fail-515d409a161f)<br>
<i class="fa fa-arrow-right" aria-hidden="true"></i> [The New Business of AI (and How It’s Different From Traditional Software)](https://a16z.com/2020/02/16/the-new-business-of-ai-and-how-its-different-from-traditional-software/){:target="_blank"}

**Bu çevirinin ve çevirideki grafiklerin izinsiz ve kaynak gösterilmeden kullanılması yasaktır.**
