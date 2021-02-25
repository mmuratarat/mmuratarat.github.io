---
layout: post
title:  "[TR] TensorFlow IO'dan PostgreSQL veritabanını okuma"
author: "MMA"
comments: true
tags: [TensorFlow, TensorFlow-IO, PostgreSQL, Turkish]
---

[PostgreSQL ve pgAdmin4 Grafik Ara yüzü Kurulumu](https://mmuratarat.github.io/2020-11-18/TR_how_to_install_postgresql_pgadmin4){:target="_blank"} isimli yazımda kişisel bilgisayarınızda `localhost` üzerinde nasıl kendi PostgreSQL sunucunuzu yaratacağınızdan ve kendi veri tabanlarınızı oluşturup, bu veri tabanlarına nasıl veri yükleyeceğinizden bahsetmiştim. Bu kısa eğiticide ise  kişisel bilgisayarınızdaki veri tabanına verilerinizi girip, Python ortamına bu verileri nasıl aktaracağınızı TensorFlow ve [TensorFlow-IO](https://www.tensorflow.org/io){:target="_blank"} kütüphanelerini kullanarak göstereceğim. Yaparak öğrenme veri biliminde kendinizi geliştirmeniz için en iyi yöntemdir. Artık bir veri tabanından verilerinizi çekip tf.data.Dataset kullanarak elde edeceğiniz veri iletim hattını, eğitim veya çıkarsama amacıyla doğrudan tf.keras’a aktararak istediğiniz derin öğrenme algoritmasını uygulayabilirsiniz.

İlk olarak gerekli kütüphaneleri içe aktararak ise başlayalım.

```python
import os
import tensorflow as tf
print("TF Version: ", tf.__version__)
#TF Version:  2.4.1
```

Burada kullanacağımız kütüphanelerden biri `tensorflow-io`. Bu nedenle bu kütüphane komut penceresinde `pip3 install tensorflow-io` komutuyla yüklenmesi gerekmektedir. JupyterLab not defteri üzerinde `!pip3 install tensorflow-io` komutunu da kullanabilirsiniz. Daha sonra bu kütüphaneyi de içer aktaralım:

```python
import tensorflow_io as tfio
```

Demo amaçlı bu eğitim, bir veritabanı oluşturacak ve veritabanını bazı verilerle dolduracaktır. Bu eğiticide kullanılan veriler [UCI Makine Öğrenmesi Deposundan (UCI Machine Learning Repository)](http://archive.ics.uci.edu/ml){:target="_blank"} indirebileceğiniz [Hava Kalitesi Veri Kümesidir (Air Quality Dataset)](https://archive.ics.uci.edu/ml/datasets/Air+Quality){:target="_blank"}.

Bu veri setini öncelikle kişisel bilgisayarınızda kurulu olan PostgreSQL veri tabanına eklemeniz gerekmektedir. Tabloyu  `CREATE TABLE` komutuyla oluşturunuz ve değişkenleri tanımlayınız (bir veri tabanında tablo yaratma ile ilgili daha fazla bilgi için [buraya](https://www.postgresqltutorial.com/postgresql-create-table/){:target="_blank"} tıklayınız.):

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/ss1_psql.png?raw=true)

```sql
CREATE TABLE IF NOT EXISTS AirQualityUCI (
  Date DATE, 
  Time TIME, 
  CO REAL,
  PT08S1 INT,
  NMHC REAL,
  C6H6 REAL,
  PT08S2 INT,
  NOx REAL,
  PT08S3 INT,
  NO2 REAL,
  PT08S4 INT,
  PT08S5 INT,
  T REAL,
  RH REAL,
  AH REAL
);
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/ss2_psql.png?raw=true)

Daha sonra bu CSV dosyasında bulunan tüm verileri PostgreSQL veri tabanına aşağıdaki komutu kullanarak girmeniz gerekmektedir:

```sql
INSERT INTO AirQualityUCI (Date, Time, CO, PT08S1, NMHC, C6H6, PT08S2, NOx, PT08S3, NO2, PT08S4, PT08S5, T, RH, AH) VALUES('2004/03/10', '18:00:00', 2.6, 1360, 150, 11.9, 1046, 166, 1056, 113, 1692, 1268, 13.6, 48.9, 0.7578);
INSERT INTO AirQualityUCI (Date, Time, CO, PT08S1, NMHC, C6H6, PT08S2, NOx, PT08S3, NO2, PT08S4, PT08S5, T, RH, AH) VALUES('2004/03/10', '19:00:00', 2, 1292, 112, 9.4, 955, 103, 1174, 92, 1559, 972, 13.3, 47.7, 0.7255);
INSERT INTO AirQualityUCI (Date, Time, CO, PT08S1, NMHC, C6H6, PT08S2, NOx, PT08S3, NO2, PT08S4, PT08S5, T, RH, AH) VALUES('2004/03/10', '20:00:00', 2.2, 1402, 88, 9.0, 939, 131, 1140, 114, 1555, 1074, 11.9, 54.0, 0.7502);
INSERT INTO AirQualityUCI (Date, Time, CO, PT08S1, NMHC, C6H6, PT08S2, NOx, PT08S3, NO2, PT08S4, PT08S5, T, RH, AH) VALUES('2004/03/10', '21:00:00', 2.2, 1376, 80, 9.2, 948, 172, 1092, 122, 1584, 1203, 11.0, 60.0, 0.7867);
INSERT INTO AirQualityUCI (Date, Time, CO, PT08S1, NMHC, C6H6, PT08S2, NOx, PT08S3, NO2, PT08S4, PT08S5, T, RH, AH) VALUES('2004/03/10', '22:00:00', 1.6, 1272, 51, 6.5, 836, 131, 1205, 116, 1490, 1110, 11.2, 59.6, 0.7888);
INSERT INTO AirQualityUCI (Date, Time, CO, PT08S1, NMHC, C6H6, PT08S2, NOx, PT08S3, NO2, PT08S4, PT08S5, T, RH, AH) VALUES('2004/03/10', '23:00:00', 1.2, 1197, 38, 4.7, 750, 89, 1337, 96, 1393, 949, 11.2, 59.2, 0.7848);
.
.
.
.
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/ss3_psql.png?raw=true)

Veri girişi tamamlandıktan sonra pgAdmin 4 kullanarak sorgulama yaptığınızda veri setini görmeniz gerekmektedir:

```sql
SELECT co, pt08s1 FROM AirQualityUCI;
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/ss4_psql.png?raw=true)

Veri hazırlamayı basitleştirmeye yardımcı olmak için, Hava Kalitesi Veri Kümesinin bir sql versiyonu hazırlanmıştır ve bu `.sql` formatlı dosyayı kullanarak da yukarıdaki tüm işlemleri tek bir adımda gerçekleştirebilirsiniz. Dosya [AirQualityUCI.sql](https://github.com/tensorflow/io/blob/master/docs/tutorials/postgresql/AirQualityUCI.sql){:target="_blank"} olarak hazırdır.

Peki bu sql dosyasını kullanarak yukarıda yaptığımız tüm işlemleri tek bir adımda nasıl gerçekleştirebiliriz?

İlk olarak yapmanız gereken `curl` kullanarak `AirQualityUCI.sql` dosyasını bilgisayarınızda istediğiniz dizine indirmek:

```bash
curl -s -OL https://github.com/tensorflow/io/raw/master/docs/tutorials/postgresql/AirQualityUCI.sql
```

Daha sonra komut satırı penceresinde `psql` kullanarak bu `.sql` dosyasını PostgreSQL'e okutabilirsiniz. `psql`, PostgreSQL için terminal tabanlı bir ön uçtur.
`AirQualityUCI.sql` dosyasını incelediğinizde, bu dosyanın veri tabanınızda yeni bir tablo yaratacağını, `AirQualityUCI.csv` dosyasından değerleri okuyup, bu tabloya bu değerleri gireceğini kolaylıkla anlayabilirsiniz.

`AirQualityUCI.sql` dosyasını bilgisayarınıza `curl` kullanarak indirdikten sonra aşağıdaki komutu kullanarak bu `.sql` dosyasini okutabilirsiniz:

```
psql -h localhost -d postgres -U postgres -p 5432 -a -q -f /Users/mustafamuratarat/AirQualityUCI.sql
```

Bu komutta bulunan her parametrenin kullanım amacı şu şekildedir:

* `-h` : PostgreSQL sunucusunun IP adresi
* `-d` : Veri tabanının ismi
* `-U` : Kullanıcı ismi
* `-p` : PostgreSQL sunucusunun dinlediği port (bağlantı noktası)
* `-f` : SQL betiğinin bulunduğu yol

Komutu çalıştırdığınızda sizden şifrenizi girmeniz istenecektir. Kullanıcı adınıza ait şifreyi girdikten sonra tablonuz veri tabanınızda oluşacak ve veriler tabloya eklenecektir. 

Bu yöntemi de gördükten sonra JupyterLab not defteri üzerinden bu veriyi PostgreSQL'e aktarma işlemine devam edelim. İlk olarak bu `.sql` dosyasının okutulması için gerekli ortam değişkenlerinin kurulması gerekmektedir. Aşağıdaki değişkenlerin değerlerini kendi bilgisayarınızda kurduğunuz PostgreSQL sunucusuna ve veri tabanına göre düzeltmeniz gerekmektedir.

```python
%env DATABASE_NAME=postgres
%env DATABASE_HOST=localhost
%env DATABASE_PORT=5432
%env DATABASE_USER=postgres
%env DATABASE_PASS=<PASSWORD>
```

Burada gerekli tüm değişkenlere sisteminizde tanımladığınız değerleri vermeniz gerekmektedir. Ben PostgreSQL kurulumunda varsayılan değerleri kullandığım için `DATABASE_NAME`, `DATABASE_HOST`, `DATABASE_PORT`, ve `DATABASE_USER` aynı olabilir. Ancak, `DATABASE_PASS` ortam değişkenine ait değer olan `<PASSWORD>` kullanıcı adınıza ait sizin belirlediğiniz şifredir.

PostgreSQL sunucusundan bir `Dataset` (Veri Kümesi) oluşturmak, `query` (sorgu) ve `endpoint` (bitis noktasi) argümanları ile `tfio.experimental.IODataset.from_sql`'yi çağırmak kadar kolaydır. `query`, tablolardaki seçilen sütunlar için SQL sorgusudur ve `endpoint` argümanı, adres ve veritabanı adıdır:

```python
endpoint="postgresql://{}:{}@{}?port={}&dbname={}".format(
    os.environ['DATABASE_USER'],
    os.environ['DATABASE_PASS'],
    os.environ['DATABASE_HOST'],
    os.environ['DATABASE_PORT'],
    os.environ['DATABASE_NAME'],
)

#endpoint
#'postgresql://postgres:<PASSWORD>@localhost?port=5432&dbname=postgres'
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/tensorflowio_endpoint.jpeg?raw=true)

```python
dataset = tfio.experimental.IODataset.from_sql(
    query="SELECT pt08s1, nmhc, c6h6, co FROM AirQualityUCI;",
    endpoint=endpoint)

print(dataset.element_spec)
#{'pt08s1': TensorSpec(shape=(), dtype=tf.int32, name=None), 'nmhc': TensorSpec(shape=(), dtype=tf.float32, name=None), 'c6h6': TensorSpec(shape=(), dtype=tf.float32, name=None), 'co': TensorSpec(shape=(), dtype=tf.float32, name=None)}
```

Yukarıdaki `dataset.element_spec` çıktısından da görebileceğiniz gibi, oluşturulan `Dataset`'in elemanı, veritabanı tablosunun sütun isimlerini anahtar olarak içeren bir python `dict` (sozluk) nesnesidir. Daha fazla işlem yapmak oldukça mümkündür.

`pt08s1`, `nmhc`, `c6h6`, ve `co`  kolonlarının olduğu ilk 10 kaydı kontrol edelim:

```
dataset = dataset.take(10)
#<TakeDataset shapes: {pt08s1: (), nmhc: (), c6h6: (), co: ()}, types: {pt08s1: tf.int32, nmhc: tf.float32, c6h6: tf.float32, co: tf.float32}>
for i in dataset.as_numpy_iterator():
    print(i)

#ya da sadece list(dataset.as_numpy_iterator()) kullanabilirsiniz.

# {'pt08s1': 1360, 'nmhc': 150.0, 'c6h6': 11.9, 'co': 2.6}
# {'pt08s1': 1292, 'nmhc': 112.0, 'c6h6': 9.4, 'co': 2.0}
# {'pt08s1': 1402, 'nmhc': 88.0, 'c6h6': 9.0, 'co': 2.2}
# {'pt08s1': 1376, 'nmhc': 80.0, 'c6h6': 9.2, 'co': 2.2}
# {'pt08s1': 1272, 'nmhc': 51.0, 'c6h6': 6.5, 'co': 1.6}
# {'pt08s1': 1197, 'nmhc': 38.0, 'c6h6': 4.7, 'co': 1.2}
# {'pt08s1': 1185, 'nmhc': 31.0, 'c6h6': 3.6, 'co': 1.2}
# {'pt08s1': 1136, 'nmhc': 31.0, 'c6h6': 3.3, 'co': 1.0}
# {'pt08s1': 1094, 'nmhc': 24.0, 'c6h6': 2.3, 'co': 0.9}
# {'pt08s1': 1010, 'nmhc': 19.0, 'c6h6': 1.7, 'co': 0.6}
```

Oluşturulan `Dataset`, eğitim veya çıkarsama amacıyla doğrudan `tf.keras`'a aktarılmaya hazırdır.


$$
x_{ij} = \sum_{i=1}^{n} x_{i}
$$
