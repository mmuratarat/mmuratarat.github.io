---
layout: post
title:  "[TR] Model Aramaya Yönelik Uygulamalı Kılavuz: AutoML için Tensorflow tabanlı bir Yazılım İskeleti"
author: "MMA"
comments: true
tags: [TensorFlow, AutoML, Turkish]
---

Derin Sinir ağları söz konusu olduğunda, belirli bir problem için uygun mimariyi (katman türleri, katman sayısı, optimizasyon türü vb.) seçmek zor ve yorucu bir süreç olabilir. Bu nedenle, son yıllarda Otomatik makine öğrenmesi (AutoML) derin öğrenme alanındaki en sıcak araştırma alanlarından biri haline geldi. AutoML, makine öğrenmesi modellerinin oluşturulmasını otomatikleştirmek için  makine öğrenmesi kullanma fikridir. Bu nedenle, Google, araştırmacıların makine öğrenimi modellerini verimli ve otomatik olarak geliştirmelerine yardımcı olmak için [AutoML iskeletini](https://ai.googleblog.com/2021/02/introducing-model-search-open-source.html){:target="_blank"} [açık kaynaklı](https://github.com/google/model_search){:target="_blank"} hale getirdi. Herhangi bir sınıflandırma problemi için doğru model mimarisini hızlı ve uygun maliyetli bir şekilde otomatik olarak bulur. Bu yazılım iskeleti, uygun hiperparametreleri bulmak için Bayes optimizasyonunu kullanır ve modellerin bir topluluğunu (ensemble) oluşturabilir. Hem yapılandırılmış hem de görüntü verileri için çalışır.

### Sanal Ortam Oluşturmak

Bu eğiticiye başlamadan önce bilgisayarınızdaki düzeni bozmamak adına, aşağıda yapacağımız tüm işlemleri sanal bir ortam içerisinde gerçekleştirelim. Basitçe ifade etmek gerekirse, bir sanal ortam, diğer projeleri etkileme endişesi olmadan belirli bir proje üzerinde çalışmanıza izin veren, Python'un yalıtılmış bir çalışma kopyasıdır. Her proje için birden çok Python versiyonunun aynı makineye kurulumuna olanak tanır. Aslında Python'un ayrı kopyalarını kurmaz, ancak ortam değişkenlerini ve paketleri izole ettiği için farklı proje ortamlarını izole tutmanın akıllıca bir yolunu sağlar. Yanlış paket versiyonlarindan şikayet eden hata mesajlarının bir çaresidir (Örneğin, kişisel bilgisayarımin global ortamında TensorFlow 2.4.1 kuruludur ancak bu eğiticide kullanacağımız `model_search` paketi TensorFlow'un 2.2.0 versiyonunu gerektirmektedir - bakınız [requirements.txt](https://github.com/google/model_search/blob/master/requirements.txt){:target="_blank"}).

Python'da sanal ortam oluşturmak için kullanabileceğiniz bazı popüler kütüphaneler/araçlar şöyledir: `virtualenv`, `virtualenvwrapper`, `pvenv` ve `venv`. Burada `virtualenv` paketine odaklanacağız.

`virtualenv`'in sisteminizde zaten kurulu olması muhtemeldir. Bununla birlikte, bu paketin global sisteminizde kurulu olup olmadığını, eğer kurulu ise, hangi sürümü kullandığınızı kontrol edin:

```
which virtualenv
```

veya

```
virtualenv --version
```
Eğer bu paket kurulu değilse, `virtualenv` paketini yüklemeniz gerekir:

```bash
Arat-MacBook-Pro:~ mustafamuratarat$ pip3 install virtualenv
Collecting virtualenv
  Downloading virtualenv-20.4.2-py2.py3-none-any.whl (7.2 MB)
     |████████████████████████████████| 7.2 MB 4.4 MB/s 
Collecting filelock<4,>=3.0.0
  Downloading filelock-3.0.12-py3-none-any.whl (7.6 kB)
Requirement already satisfied: six<2,>=1.9.0 in /Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages (from virtualenv) (1.15.0)
Collecting appdirs<2,>=1.4.3
  Downloading appdirs-1.4.4-py2.py3-none-any.whl (9.6 kB)
Collecting distlib<1,>=0.3.1
  Downloading distlib-0.3.1-py2.py3-none-any.whl (335 kB)
     |████████████████████████████████| 335 kB 4.8 MB/s 
Installing collected packages: filelock, distlib, appdirs, virtualenv
Successfully installed appdirs-1.4.4 distlib-0.3.1 filelock-3.0.12 virtualenv-20.4.2
Arat-MacBook-Pro:~ mustafamuratarat$ which virtualenv
/Library/Frameworks/Python.framework/Versions/3.8/bin/virtualenv
Arat-MacBook-Pro:~ mustafamuratarat$ virtualenv --version
virtualenv 20.4.2 from /Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/virtualenv/__init__.py
```

Bunu işlemi gerçekleştirdikten sonra, uygulamanızın yer alacağı klasöre gidin (ben bugünki eğitici için `tutorial1` isminde yeni bir klasör yarattım) ve aşağıda verilenleri gerçekleştirin. Aşağıdaki kod, kurduğumuz `virtualenv` modülünü çağıracak ve mevcut klasörümüzün içinde `venv` adlı yeni bir klasör oluşturacak (yani `venv` isminde bir sanal ortam) ve `venv`'de yeni bir Python kurulumu yükleyecek (tabii ki kurmak istediğiniz Python sürümünü değiştirebilirsiniz). Ben burada sistemimde kurulu olan aynı Python 3 sürümünü sanal ortamımda da kullanmak istediğim için `which python3` komutunu kullandım:

```
virtualenv venv -p `which python3`
```

Yukarıdaki komut, projenizde tüm bağımlılıkların (dependencies) kurulu olduğu bir `venv/` dizini oluşturur. Kurulum tamamlandıktan sonra, bu sanal ortamı kullanmak istiyorsanız, izole edilmiş ortamınızı etkinleştirmeniz gerekir:

```
Arat-MacBook-Pro:tutorial1 mustafamuratarat$ source venv/bin/activate
(venv) Arat-MacBook-Pro:tutorial1 mustafamuratarat$ 
```

Burada dikkat etmeniz gereken nokta, komut satırının bilgisayarınızın adından, yeni oluşturduğunuz sanal ortamın ismine dönüşmesidir. Proje üzerinde her çalışmak istediğinizde bu sanal ortamı aktive etmelisiniz. Bu nedenle `source venv/bin/activate` kodunu çalıştırmayı unutmayın. Ancak, her seferinde bu kodu yazmak istemediğinizde ve bilgisayarınız başlar başlamaz, sanal ortamınızın aktive olmasını istiyorsanız [şu sayfada](https://askubuntu.com/a/1175106/1187527){:target="_blank"} bulunan adımları takip ederek bir betik yazabilirsiniz.

Artık bu ortamda istediğiniz paketleri ve bu paketlerin versiyonlarını global sisteminizi etkilemeden kurabilirsiniz. Burada Google'un Github sayfasında bulunan `model_search` paketini kullanacağımız için, bu paket için gerekli tüm paketleri ve bu paketlerin versiyonlarını [requirements.txt](https://github.com/google/model_search/blob/master/requirements.txt){:target="_blank"} dosyasında bulabilirsiniz:

`model_search` paketi, PyPI'de henüz mevcut değildir, bu nedenle `git` kullanılarak klonlanabilir. Klonlama işlemi terminal penceresinde sanal ortamımızda yapılır:

```
(venv) Arat-MacBook-Pro:tutorial1 mustafamuratarat$ git clone https://github.com/google/model_search.git
Cloning into 'model_search'...
remote: Enumerating objects: 141, done.
remote: Counting objects: 100% (141/141), done.
remote: Compressing objects: 100% (109/109), done.
remote: Total 141 (delta 30), reused 136 (delta 28), pack-reused 0
Receiving objects: 100% (141/141), 214.12 KiB | 1.06 MiB/s, done.
Resolving deltas: 100% (30/30), done.
```

Şimdi, `tutorial1` klasörünün altında `model_search` isminde yeni bir klasör oluşacak. `cd` komutunu kullanarak `model_search` klasörüne gidelim ve `requirements.txt` dosyasını çalıştırarak, gerekli tüm modülleri sanal ortamımızdaki Python için yükleyelim.

```
(venv) Arat-MacBook-Pro:tutorial1 mustafamuratarat$ cd model_search/
(venv) Arat-MacBook-Pro:model_search mustafamuratarat$ pip install -r requirements.txt
Collecting six==1.15.0
  Downloading six-1.15.0-py2.py3-none-any.whl (10 kB)
Collecting sklearn==0.0
  Downloading sklearn-0.0.tar.gz (1.1 kB)
Collecting tensorflow==2.2.0
  Downloading tensorflow-2.2.0-cp38-cp38-macosx_10_11_x86_64.whl (175.4 MB)
     |████████████████████████████████| 175.4 MB 4.7 MB/s 
Collecting absl-py==0.10.0
  Downloading absl_py-0.10.0-py3-none-any.whl (127 kB)
     |████████████████████████████████| 127 kB 5.1 MB/s 
Collecting tf-slim==1.1.0
  Downloading tf_slim-1.1.0-py2.py3-none-any.whl (352 kB)
     |████████████████████████████████| 352 kB 4.3 MB/s 
Collecting ml-metadata==0.26.0
  Downloading ml_metadata-0.26.0-cp38-cp38-macosx_10_9_x86_64.whl (5.3 MB)
     |████████████████████████████████| 5.3 MB 4.1 MB/s 
Collecting keras-tuner==1.0.2
  Downloading keras-tuner-1.0.2.tar.gz (62 kB)
     |████████████████████████████████| 62 kB 737 kB/s 
Collecting mock==4.0.3
  Downloading mock-4.0.3-py3-none-any.whl (28 kB)
Collecting packaging
  Downloading packaging-20.9-py2.py3-none-any.whl (40 kB)
     |████████████████████████████████| 40 kB 3.4 MB/s 
Collecting future
  Downloading future-0.18.2.tar.gz (829 kB)
     |████████████████████████████████| 829 kB 4.0 MB/s 
Collecting numpy
  Downloading numpy-1.20.1-cp38-cp38-macosx_10_9_x86_64.whl (16.0 MB)
     |████████████████████████████████| 16.0 MB 1.8 MB/s 
Collecting tabulate
  Downloading tabulate-0.8.9-py3-none-any.whl (25 kB)
Collecting terminaltables
  Downloading terminaltables-3.1.0.tar.gz (12 kB)
Collecting colorama
  Downloading colorama-0.4.4-py2.py3-none-any.whl (16 kB)
Collecting tqdm
  Downloading tqdm-4.58.0-py2.py3-none-any.whl (73 kB)
     |████████████████████████████████| 73 kB 2.2 MB/s 
Collecting requests
  Downloading requests-2.25.1-py2.py3-none-any.whl (61 kB)
     |████████████████████████████████| 61 kB 3.5 MB/s 
Collecting scipy
  Downloading scipy-1.6.1-cp38-cp38-macosx_10_9_x86_64.whl (30.8 MB)
     |████████████████████████████████| 30.8 MB 6.0 MB/s 
Collecting scikit-learn
  Downloading scikit_learn-0.24.1-cp38-cp38-macosx_10_13_x86_64.whl (7.2 MB)
     |████████████████████████████████| 7.2 MB 7.1 MB/s 
Collecting grpcio<2,>=1.8.6
  Downloading grpcio-1.36.0-cp38-cp38-macosx_10_10_x86_64.whl (3.8 MB)
     |████████████████████████████████| 3.8 MB 198 kB/s 
Collecting attrs<21,>=20.3
  Downloading attrs-20.3.0-py2.py3-none-any.whl (49 kB)
     |████████████████████████████████| 49 kB 2.3 MB/s 
Collecting protobuf<4,>=3.7
  Downloading protobuf-3.15.3-cp38-cp38-macosx_10_9_x86_64.whl (1.0 MB)
     |████████████████████████████████| 1.0 MB 3.0 MB/s 
Collecting keras-preprocessing>=1.1.0
  Downloading Keras_Preprocessing-1.1.2-py2.py3-none-any.whl (42 kB)
     |████████████████████████████████| 42 kB 1.4 MB/s 
Collecting scipy
  Downloading scipy-1.4.1-cp38-cp38-macosx_10_9_x86_64.whl (28.8 MB)
     |████████████████████████████████| 28.8 MB 1.6 MB/s 
Collecting tensorflow-estimator<2.3.0,>=2.2.0
  Downloading tensorflow_estimator-2.2.0-py2.py3-none-any.whl (454 kB)
     |████████████████████████████████| 454 kB 992 kB/s 
Collecting h5py<2.11.0,>=2.10.0
  Downloading h5py-2.10.0-cp38-cp38-macosx_10_9_x86_64.whl (3.0 MB)
     |████████████████████████████████| 3.0 MB 1.9 MB/s 
Collecting tensorboard<2.3.0,>=2.2.0
  Downloading tensorboard-2.2.2-py3-none-any.whl (3.0 MB)
     |████████████████████████████████| 3.0 MB 1.8 MB/s 
Collecting wrapt>=1.11.1
  Downloading wrapt-1.12.1.tar.gz (27 kB)
Collecting google-pasta>=0.1.8
  Downloading google_pasta-0.2.0-py3-none-any.whl (57 kB)
     |████████████████████████████████| 57 kB 2.2 MB/s 
Collecting termcolor>=1.1.0
  Downloading termcolor-1.1.0.tar.gz (3.9 kB)
Collecting opt-einsum>=2.3.2
  Downloading opt_einsum-3.3.0-py3-none-any.whl (65 kB)
     |████████████████████████████████| 65 kB 2.4 MB/s 
Requirement already satisfied: wheel>=0.26 in /Users/mustafamuratarat/tutorial1/venv/lib/python3.8/site-packages (from tensorflow==2.2.0->-r requirements.txt (line 3)) (0.36.2)
Collecting gast==0.3.3
  Downloading gast-0.3.3-py2.py3-none-any.whl (9.7 kB)
Collecting astunparse==1.6.3
  Downloading astunparse-1.6.3-py2.py3-none-any.whl (12 kB)
Collecting tensorboard-plugin-wit>=1.6.0
  Downloading tensorboard_plugin_wit-1.8.0-py3-none-any.whl (781 kB)
     |████████████████████████████████| 781 kB 2.4 MB/s 
Collecting werkzeug>=0.11.15
  Downloading Werkzeug-1.0.1-py2.py3-none-any.whl (298 kB)
     |████████████████████████████████| 298 kB 2.5 MB/s 
Collecting markdown>=2.6.8
  Downloading Markdown-3.3.4-py3-none-any.whl (97 kB)
     |████████████████████████████████| 97 kB 2.6 MB/s 
Requirement already satisfied: setuptools>=41.0.0 in /Users/mustafamuratarat/tutorial1/venv/lib/python3.8/site-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0->-r requirements.txt (line 3)) (52.0.0)
Collecting google-auth<2,>=1.6.3
  Downloading google_auth-1.27.0-py2.py3-none-any.whl (135 kB)
     |████████████████████████████████| 135 kB 2.7 MB/s 
Collecting google-auth-oauthlib<0.5,>=0.4.1
  Downloading google_auth_oauthlib-0.4.2-py2.py3-none-any.whl (18 kB)
Collecting pyasn1-modules>=0.2.1
  Downloading pyasn1_modules-0.2.8-py2.py3-none-any.whl (155 kB)
     |████████████████████████████████| 155 kB 2.5 MB/s 
Collecting cachetools<5.0,>=2.0.0
  Downloading cachetools-4.2.1-py3-none-any.whl (12 kB)
Collecting rsa<5,>=3.1.4
  Downloading rsa-4.7.2-py3-none-any.whl (34 kB)
Collecting requests-oauthlib>=0.7.0
  Downloading requests_oauthlib-1.3.0-py2.py3-none-any.whl (23 kB)
Collecting pyasn1<0.5.0,>=0.4.6
  Downloading pyasn1-0.4.8-py2.py3-none-any.whl (77 kB)
     |████████████████████████████████| 77 kB 2.4 MB/s 
Collecting urllib3<1.27,>=1.21.1
  Downloading urllib3-1.26.3-py2.py3-none-any.whl (137 kB)
     |████████████████████████████████| 137 kB 2.9 MB/s 
Collecting certifi>=2017.4.17
  Downloading certifi-2020.12.5-py2.py3-none-any.whl (147 kB)
     |████████████████████████████████| 147 kB 2.6 MB/s 
Collecting chardet<5,>=3.0.2
  Downloading chardet-4.0.0-py2.py3-none-any.whl (178 kB)
     |████████████████████████████████| 178 kB 2.6 MB/s 
Collecting idna<3,>=2.5
  Downloading idna-2.10-py2.py3-none-any.whl (58 kB)
     |████████████████████████████████| 58 kB 2.3 MB/s 
Collecting oauthlib>=3.0.0
  Downloading oauthlib-3.1.0-py2.py3-none-any.whl (147 kB)
     |████████████████████████████████| 147 kB 2.4 MB/s 
Collecting pyparsing>=2.0.2
  Downloading pyparsing-2.4.7-py2.py3-none-any.whl (67 kB)
     |████████████████████████████████| 67 kB 2.3 MB/s 
Collecting threadpoolctl>=2.0.0
  Downloading threadpoolctl-2.1.0-py3-none-any.whl (12 kB)
Collecting joblib>=0.11
  Downloading joblib-1.0.1-py3-none-any.whl (303 kB)
     |████████████████████████████████| 303 kB 2.8 MB/s 
Building wheels for collected packages: keras-tuner, sklearn, termcolor, wrapt, future, terminaltables
  Building wheel for keras-tuner (setup.py) ... done
  Created wheel for keras-tuner: filename=keras_tuner-1.0.2-py3-none-any.whl size=78936 sha256=02dcb8f0ad19bd6ea51904382b98f5ab6d3bda1cad8dcae399c05b1fb17e45fb
  Stored in directory: /Users/mustafamuratarat/Library/Caches/pip/wheels/53/3d/c3/160c686bd74a18989843fcd015e8f6954ca8d834fd2ef4658a
  Building wheel for sklearn (setup.py) ... done
  Created wheel for sklearn: filename=sklearn-0.0-py2.py3-none-any.whl size=1316 sha256=a4e68b09dcf036080df23f7c48b1691fba5eb55645557ebde5c7638a350fb3c6
  Stored in directory: /Users/mustafamuratarat/Library/Caches/pip/wheels/22/0b/40/fd3f795caaa1fb4c6cb738bc1f56100be1e57da95849bfc897
  Building wheel for termcolor (setup.py) ... done
  Created wheel for termcolor: filename=termcolor-1.1.0-py3-none-any.whl size=4829 sha256=e0479dae834087e86a493b048ba31295161b8f015e741cabb4d19713ad327444
  Stored in directory: /Users/mustafamuratarat/Library/Caches/pip/wheels/a0/16/9c/5473df82468f958445479c59e784896fa24f4a5fc024b0f501
  Building wheel for wrapt (setup.py) ... done
  Created wheel for wrapt: filename=wrapt-1.12.1-cp38-cp38-macosx_10_9_x86_64.whl size=32639 sha256=b525b6145b5a1de40a8a803d2106c55369b484e6d8f91a2bec8ed3564b5772cb
  Stored in directory: /Users/mustafamuratarat/Library/Caches/pip/wheels/5f/fd/9e/b6cf5890494cb8ef0b5eaff72e5d55a70fb56316007d6dfe73
  Building wheel for future (setup.py) ... done
  Created wheel for future: filename=future-0.18.2-py3-none-any.whl size=491059 sha256=d2edd4702e13ee3cc03c147f7eedc5d632ffb594b603892d2cb95f5a4599c155
  Stored in directory: /Users/mustafamuratarat/Library/Caches/pip/wheels/8e/70/28/3d6ccd6e315f65f245da085482a2e1c7d14b90b30f239e2cf4
  Building wheel for terminaltables (setup.py) ... done
  Created wheel for terminaltables: filename=terminaltables-3.1.0-py3-none-any.whl size=15355 sha256=6e7c30e0f5c78f140c20a4709b558bdf198a966ebbd2ee0ec9ba983d2bd40d62
  Stored in directory: /Users/mustafamuratarat/Library/Caches/pip/wheels/08/8f/5f/253d0105a55bd84ee61ef0d37dbf70421e61e0cd70cef7c5e1
Successfully built keras-tuner sklearn termcolor wrapt future terminaltables
Installing collected packages: urllib3, pyasn1, idna, chardet, certifi, six, rsa, requests, pyasn1-modules, oauthlib, cachetools, requests-oauthlib, numpy, google-auth, werkzeug, threadpoolctl, tensorboard-plugin-wit, scipy, pyparsing, protobuf, markdown, joblib, grpcio, google-auth-oauthlib, absl-py, wrapt, tqdm, terminaltables, termcolor, tensorflow-estimator, tensorboard, tabulate, scikit-learn, packaging, opt-einsum, keras-preprocessing, h5py, google-pasta, gast, future, colorama, attrs, astunparse, tf-slim, tensorflow, sklearn, mock, ml-metadata, keras-tuner
Successfully installed absl-py-0.10.0 astunparse-1.6.3 attrs-20.3.0 cachetools-4.2.1 certifi-2020.12.5 chardet-4.0.0 colorama-0.4.4 future-0.18.2 gast-0.3.3 google-auth-1.27.0 google-auth-oauthlib-0.4.2 google-pasta-0.2.0 grpcio-1.36.0 h5py-2.10.0 idna-2.10 joblib-1.0.1 keras-preprocessing-1.1.2 keras-tuner-1.0.2 markdown-3.3.4 ml-metadata-0.26.0 mock-4.0.3 numpy-1.20.1 oauthlib-3.1.0 opt-einsum-3.3.0 packaging-20.9 protobuf-3.15.3 pyasn1-0.4.8 pyasn1-modules-0.2.8 pyparsing-2.4.7 requests-2.25.1 requests-oauthlib-1.3.0 rsa-4.7.2 scikit-learn-0.24.1 scipy-1.4.1 six-1.15.0 sklearn-0.0 tabulate-0.8.9 tensorboard-2.2.2 tensorboard-plugin-wit-1.8.0 tensorflow-2.2.0 tensorflow-estimator-2.2.0 termcolor-1.1.0 terminaltables-3.1.0 tf-slim-1.1.0 threadpoolctl-2.1.0 tqdm-4.58.0 urllib3-1.26.3 werkzeug-1.0.1 wrapt-1.12.1
```

Aşağıdaki eğitici JupyterLab üzerinden gerçekleştirilecektir. Ancak buradaki problem ise, kişisel bilgisayarınızdaki JupyterLab'ın global sisteminizde bulunan Python sürümü için çekirdeğinin (kernel) bulunmasıdır. Bu nedenle, sanal ortamımıza yüklediğimiz Python için çekirdek elle oluşturmamız için bazı işlemler daha yapmamız gerekmektedir. Bunun için ilk olarak Jupyter için IPython çekirdeğini sağlayan `ipykernel`'i sanal ortamımızda kurmamız gerekmektedir:

```
(venv) Arat-MacBook-Pro:model_search mustafamuratarat$ pip install ipykernel
Collecting ipykernel
  Downloading ipykernel-5.5.0-py3-none-any.whl (120 kB)
     |████████████████████████████████| 120 kB 781 kB/s 
Collecting traitlets>=4.1.0
  Downloading traitlets-5.0.5-py3-none-any.whl (100 kB)
     |████████████████████████████████| 100 kB 1.2 MB/s 
Collecting jupyter-client
  Downloading jupyter_client-6.1.11-py3-none-any.whl (108 kB)
     |████████████████████████████████| 108 kB 1.3 MB/s 
Collecting tornado>=4.2
  Downloading tornado-6.1-cp38-cp38-macosx_10_9_x86_64.whl (416 kB)
     |████████████████████████████████| 416 kB 1.3 MB/s 
Collecting appnope
  Downloading appnope-0.1.2-py2.py3-none-any.whl (4.3 kB)
Collecting ipython>=5.0.0
  Downloading ipython-7.21.0-py3-none-any.whl (784 kB)
     |████████████████████████████████| 784 kB 1.5 MB/s 
Collecting backcall
  Downloading backcall-0.2.0-py2.py3-none-any.whl (11 kB)
Collecting pickleshare
  Downloading pickleshare-0.7.5-py2.py3-none-any.whl (6.9 kB)
Collecting pexpect>4.3
  Downloading pexpect-4.8.0-py2.py3-none-any.whl (59 kB)
     |████████████████████████████████| 59 kB 877 kB/s 
Collecting decorator
  Downloading decorator-4.4.2-py2.py3-none-any.whl (9.2 kB)
Requirement already satisfied: setuptools>=18.5 in /Users/mustafamuratarat/tutorial1/venv/lib/python3.8/site-packages (from ipython>=5.0.0->ipykernel) (52.0.0)
Collecting jedi>=0.16
  Downloading jedi-0.18.0-py2.py3-none-any.whl (1.4 MB)
     |████████████████████████████████| 1.4 MB 881 kB/s 
Collecting pygments
  Downloading Pygments-2.8.0-py3-none-any.whl (983 kB)
     |████████████████████████████████| 983 kB 723 kB/s 
Collecting prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0
  Downloading prompt_toolkit-3.0.16-py3-none-any.whl (366 kB)
     |████████████████████████████████| 366 kB 828 kB/s 
Collecting parso<0.9.0,>=0.8.0
  Downloading parso-0.8.1-py2.py3-none-any.whl (93 kB)
     |████████████████████████████████| 93 kB 846 kB/s 
Collecting ptyprocess>=0.5
  Downloading ptyprocess-0.7.0-py2.py3-none-any.whl (13 kB)
Collecting wcwidth
  Downloading wcwidth-0.2.5-py2.py3-none-any.whl (30 kB)
Collecting ipython-genutils
  Downloading ipython_genutils-0.2.0-py2.py3-none-any.whl (26 kB)
Collecting pyzmq>=13
  Downloading pyzmq-22.0.3-cp38-cp38-macosx_10_9_x86_64.whl (1.1 MB)
     |████████████████████████████████| 1.1 MB 963 kB/s 
Collecting python-dateutil>=2.1
  Downloading python_dateutil-2.8.1-py2.py3-none-any.whl (227 kB)
     |████████████████████████████████| 227 kB 1.2 MB/s 
Collecting jupyter-core>=4.6.0
  Downloading jupyter_core-4.7.1-py3-none-any.whl (82 kB)
     |████████████████████████████████| 82 kB 783 kB/s 
Requirement already satisfied: six>=1.5 in /Users/mustafamuratarat/tutorial1/venv/lib/python3.8/site-packages (from python-dateutil>=2.1->jupyter-client->ipykernel) (1.15.0)
Installing collected packages: ipython-genutils, wcwidth, traitlets, ptyprocess, parso, tornado, pyzmq, python-dateutil, pygments, prompt-toolkit, pickleshare, pexpect, jupyter-core, jedi, decorator, backcall, appnope, jupyter-client, ipython, ipykernel
Successfully installed appnope-0.1.2 backcall-0.2.0 decorator-4.4.2 ipykernel-5.5.0 ipython-7.21.0 ipython-genutils-0.2.0 jedi-0.18.0 jupyter-client-6.1.11 jupyter-core-4.7.1 parso-0.8.1 pexpect-4.8.0 pickleshare-0.7.5 prompt-toolkit-3.0.16 ptyprocess-0.7.0 pygments-2.8.0 python-dateutil-2.8.1 pyzmq-22.0.3 tornado-6.1 traitlets-5.0.5 wcwidth-0.2.5
```

Ardından, sanal ortamınız için oluşturduğunuz Python çekirdeğini aşağıdaki komutu kullanarak Jupyter'e ekleyebilirsiniz:

```
(venv) Arat-MacBook-Pro:model_search mustafamuratarat$ python3 -m ipykernel install --user --name=venv

Installed kernelspec venv in /Users/mustafamuratarat/Library/Jupyter/kernels/venv
```

Bu klasörde, her şeyi doğru yaptıysanız aşağıdaki şekilde görünmesi gereken bir `kernel.json` dosyası bulacaksınız:

```
{
 "argv": [
  "/Users/mustafamuratarat/tutorial1/venv/bin/python3",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "venv",
 "language": "python"
}
```

Hepsi bu kadar! Artık `venv` sanal ortamını Jupyter'de çekirdek olarak seçebilirsiniz. JupyterLab'de bunun nasıl görüneceği aşağıda görülmektedir:

![](venv_kernel.png)

Yeni bir Jupyter not defteri başlattığınızda, sağ üst köşedeki daire içerisinde çekirdeğinizin ismi yazmalıdır:

![](venv_kernel_empty_Jupyternotebook.png)

Sistemimiz hazır. Şimdi Google'ın Model Search paketinin uygulamasını görebilir ve JupyterLab üzerinde çalışmaya başlayabiliriz. 

![](automl_model_search.png)

Son yıllarda, otomatikleştirilmiş makine öğrenmesi veya kısaca AutoML, araştırmacıların ve geliştiricilerin insan müdahalesi olmadan yüksek kaliteli derin öğrenme modelleri oluşturmalarına ve kullanılabilirliğini genişletmelerine gerçekten yardımcı oldu. Bu nedenle, Google, `Model Search` (Model Arama) adlı yeni bir iskelet geliştirdi.

`Model Search` (Model Arama), büyük ölçekte AutoML algoritmaları oluşturmak için açık kaynaklı, TensorFlow tabanlı bir Python iskeletidir. Bu yazılım iskeleti,

* doğru model mimarisinin araştırılmasından, en iyi damıtılmış modellere kadar birçok AutoML algoritmasını çalıştırmak için,
* arama uzayında bulunan farklı algoritmaları karşılaştırmak için, ve
* arama uzayında sinir ağı katmanlarını özelleştirmek için

kullanılır.

`Model Search` (Model Arama) fikri, Google Interspeech 2019'da, Hanna Mazzawi, Javier Gonzalvo, Aleks Kracun, Prashant Sridhar, Niranjan Subrahmanya, Ignacio Lopez Moreno, Hyun Jin Park, Patrick Violette tarafından yazılan [Improving Keyword Spotting and Language Identification via Neural Architecture Search at Scale](https://www.isca-speech.org/archive/Interspeech_2019/abstracts/1916.html){:target="_blank"} adlı çalışmayla sunuldu. `Model Search` (Model Arama)'ün ana fikri, aşağıdakileri hedefleyen yeni bir Sinir Mimarisi araştırması geliştirmektir:

* Artımlı bir arama tanımlama.
* Transfer eğitimden yararlanma.
* Genel sinir ağı bloklarının kullanımı.

<iframe width="560" height="315" src="https://www.youtube.com/embed/eptyMYo6ukw?controls=0" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

Ancak, maalesef, bu paketin halen bir kaç kısıtlaması vardır. Bu kısıtlamalardan biri, `module_search`'ün veri önişlemeyi gerçekleştirememesidir. Yani, veri temizleme ve öznitelik mühendisliği için herhangi bir iletim hattı sağlamaz. Kullanıcıların bu adımı manuel olarak yapması gerekir.

Ayrıca mevcut modül şimdilik sadece sınıflandırma problemlerini desteklemektedir.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/protobuf.png?raw=true)

Farklı dilleri konuşan insanlar bir araya gelip konuştuklarında, gruptaki herkesin anlayacağı bir dil kullanmaya çalışırlar.

Bunu başarmak için herkesin, genellikle kendi ana dilinde olan düşüncelerini grubun diline çevirmesi gerekir. Bununla birlikte, dilin bu "kodlanması (encoding) ve kodunun çözülmesi (decoding)", verimlilik, hız ve hassasiyet kaybına yol açar.

Aynı kavram bilgisayar sistemlerinde ve bileşenlerinde de mevcuttur. Doğrudan neden bahsettiklerini anlamamıza gerek yoksa verileri neden XML, JSON veya başka herhangi bir insan tarafından okunabilir formatta göndermeliyiz? Açıkça ihtiyaç duyulursa, bunu insan tarafından okunabilir bir biçime çevirebildiğimiz sürece.

[Protokol Tamponları](https://developers.google.com/protocol-buffers/?hl=en){:target="_blank"}, veri bloklarını verimli bir şekilde küçülten ve dolayısıyla bu verileri gönderirken hızı artıran, taşıma öncesi verileri kodlamanın bir yoludur. Verileri dil ve platformdan bağımsız bir biçimde özetler.

Protokol Tamponları (Protobuf), insan tarafından okunabilen formatlar olan JSON ve XML gibi, yapılandırılmış verileri serileştirme yöntemidir. Google tarafından geliştirilmiştir. Serileştirilecek verilerin tanımı, `proto` dosyaları adı verilen yapılandırma dosyalarına yazılır. Yani, bir `proto` dosyası, Google'ın Protokol Tamponu formatında oluşturulmuş geliştirici dosyasıdır.`.proto` dosyasını seçtiğimiz dil sınıfına derlemek için, proto derleyici olan `protoc`'u kullanabiliriz.

Bu nedenle bu modülün de bilgisayarınızda yüklü olması gerekmektedir.

MacOS kullanıcıları için Homebrew üzerinden aşağıdaki komutu kullanarak `protoc` paketini kişisel bilgisayarınıza kurabilirsiniz:

```bash
brew install protobuf
```

Daha sonra, `protoc` derleyicisini kullanarak tüm proto dosyalarını derleyen kod aşağıda mevcuttur. 

```python
import os
print(os.getcwd())
#/Users/mustafamuratarat/tutorial1

import pandas as pd

os.chdir('./model_search')
print(os.getcwd())
#/Users/mustafamuratarat/tutorial1/model_search

%%bash
protoc --python_out=../ model_search/proto/phoenix_spec.proto
protoc --python_out=../ model_search/proto/hparam.proto
protoc --python_out=../ model_search/proto/distillation_spec.proto
protoc --python_out=../ model_search/proto/ensembling_spec.proto
protoc --python_out=../ model_search/proto/transfer_learning_spec.proto
```

`model_search` paketini içe aktarırken herhangi bir hata (örneğin, _unparsed flag_ hatası) alırsanız [flag'leri](https://abseil.io/docs/python/guides/flags){:target="_blank"} yükleyebilirsiniz. Kod parçacığı aşağıda mevcuttur:

```python
import sys
import os
from absl import app

# Addresses `UnrecognizedFlagError: Unknown command line flag 'f'`
sys.argv = sys.argv[:1]

# `app.run` calls `sys.exit`
try:
    app.run(lambda argv: None)
except:
    pass
```

İlk olarak tüm gerekli modülleri ve paketleri içeri aktarın.

```python
import model_search
from model_search import constants
from model_search import single_trainer
from model_search.data import csv_data
```

Bu eğitici için `model_search` paketinde bunan `csv_random_data.csv` örnek verisini kullanalım.

```python
dataset = pd.read_csv("../model_search/data/testdata/csv_random_data.csv")

dataset.shape
#(20, 4)

dataset
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/dataset_modelsearch.png?raw=true)

Bu veri seti 20 gözlemden ve 3 bağımsız değişkenden oluşmaktadır. İki sınıflandırma problemi için kullanılır. İlk sütun (yani indeksi 0) sınıfların bulunduğu değişkendir.

Siz, herhangi bir sınıflandırma problemi için istediğiniz veri setini kullanabilirsiniz.

Şimdi model aramaya başlayabiliriz. İlk olarak bir `trainer` örneği (instance) oluşturup csv dosyasındaki verileri `csv_data.Provider` fonksiyonuna gönderin:

```python
trainer =  single_trainer.SingleTrainer(data=csv_data.Provider(label_index=0,
                                                              logits_dimension=2,
                                                              record_defaults=[0, 0, 0, 0],
                                                              filename="model_search/data/testdata/csv_random_data.csv"),
                                        spec= "model_search/configs/dnn_config.pbtxt") 
```

Buradaki `SingleTrainer` fonksiyonunun argümanları şu şekilde özetlenebilir: `label_index`, etiketlerin (labels) dataframe'de bulunduğu sütun numarasını gösterir. `logit_dimension`, sınıflandırma için kullanılacak verideki sınıfların sayısını temsil eder. Burada ikili sınıflandırma yaptığımız için 2 olarak değer verilmiştir. `record_default` argümanı, sütunlarda herhangi bir boş değer (null value) olduğu zaman, o değerin 0 ile değiştirilmesi gerektiğini söyleyen ve boyutu, öznitelik sayısına eşit olan diziyi temsil eder. `filename`, verilerin bulunduğu dosyanın yolunu tanımlar. Son olarak, `spec`, arama alanını temsil eder, aşağıda belirtildiği gibi kendi alanınızı oluşturabilir veya varsayılan arama alanını kullanabilirsiniz.

Model aramaya geçmeden önce `tutorial1` klasöründe sonuçların kaydedileceği bir dizin yaratalım:

```python
#os.chdir('..')
os.makedirs("../output")
```

`try_models` fonksiyonu aracılığıyla `trainer` nesnesi üzerinde farklı modelleri deneyebilirsiniz. Topluluk yöntemleri (ensemble methods) de denenecek modeller arasındadır. Aşağıdaki fonksiyonun son adımı bu yöntemin çıktısını verecektir.

```python
 trainer.try_models(
     number_models=5,
     train_steps=10,
     eval_steps=10,
     root_dir="../output/",
     batch_size=32,
     experiment_name="example",
     experiment_owner="model_search_user")
```

Burada,

* `number_models`: denenecek model sayısını temsil eder.
* `train_steps`: her modelin kaç adım için eğitilmesi gerektiğini gösterir.
* `eval_steps`: her 100 adımda modelin değerlendirilmesi gerektiğini gösterir.
* `root_dir`: sonuçların kaydedileceği dizine giden yol.
* `batch_size`: alınan veriler için yığın boyutunu temsil eder.
* `experiment_name`: deneyin adını temsil eder (ek bilgi)
* `experiment_owner`: deneyin sahibini temsil eder (ek bilgi)
 
Arama, varsayılan spesifikasyona göre yapılacaktır. Bu, `model_search/configs/dnn_config.pbtxt` dosyasında bulunabilir.

```
minimum_depth: 1
maximum_depth: 5
problem_type: DNN

search_type: ADAPTIVE_COORDINATE_DESCENT

blocks_to_use: "FIXED_OUTPUT_FULLY_CONNECTED_128"
blocks_to_use: "FIXED_OUTPUT_FULLY_CONNECTED_256"
blocks_to_use: "FIXED_OUTPUT_FULLY_CONNECTED_512"
blocks_to_use: "FIXED_OUTPUT_FULLY_CONNECTED_1024"
blocks_to_use: "FULLY_CONNECTED_RESIDUAL_FORCE_MATCH_SHAPES"
blocks_to_use: "FULLY_CONNECTED_RESIDUAL_FORCE_MATCH_SHAPES_BATCHNORM"
blocks_to_use: "FULLY_CONNECTED_RESIDUAL_CONCAT"
blocks_to_use: "FULLY_CONNECTED_RESIDUAL_CONCAT_BATCHNORM"
blocks_to_use: "FULLY_CONNECTED_RESIDUAL_PROJECT"
blocks_to_use: "FULLY_CONNECTED_RESIDUAL_PROJECT_BATCHNORM"
blocks_to_use: "FULLY_CONNECTED_PYRAMID"

apply_dropouts_between_blocks: true

beam_size: 5
increase_complexity_probability: 0.9

learning_spec {
  apply_exponential_decay: true
  apply_gradient_clipping: true
  apply_l2_regularization: false
}

ensemble_spec {
  combining_type: AVERAGE_ENSEMBLE
  ensemble_search_type: INTERMIXED_NONADAPTIVE_ENSEMBLE_SEARCH
  intermixed_search {
    width: 3
    try_ensembling_every: 5
    num_trials_to_consider: 15
  }
}

distillation_spec {
  distillation_type: ADAPTIVELY_BALANCE_LOSSES
}
```

Dosyada bulunan alanlar hakkında daha fazla ayrıntı için ve kendi spesifikasyonunuzu oluşturmak istiyorsanız `model_search/proto/phoenix_spec.proto` dosyasına bakabilirsiniz.

Arama alanında eğitime, aramaya ve değerlendirmeye başlamak için yukarıdaki kodu çalıştırın. Yukarıdaki örnek, her biri 10 adım için 5 farklı modeli dener ve modeli her 10 adımda değerlendirir. Çalışması biraz zaman alabilir.

Yukarıdaki kodun çıktısı tüm model ID'lerini, her adımdaki doğruluk oranını, aucPR/aucROC/kayıp (loss) değerlerini ve parametre sayısını içerecektir.

Gerçekleştirilen tüm denemelere göz atabilirsiniz:

```python
!ls -la ../output/example
# total 8
# drwxr-xr-x  8 mustafamuratarat  staff   256 Mar  1 08:39 .
# drwxr-xr-x  4 mustafamuratarat  staff   128 Mar  1 08:39 ..
# -rw-r--r--  1 mustafamuratarat  staff  3012 Mar  1 08:39 oracle.json
# drwxr-xr-x  3 mustafamuratarat  staff    96 Mar  1 08:39 trial_110dbfbcdb6cc359f9d1270ba17ad2e7
# drwxr-xr-x  3 mustafamuratarat  staff    96 Mar  1 08:39 trial_17c9695fe151930bc22da147cb069f8d
# drwxr-xr-x  3 mustafamuratarat  staff    96 Mar  1 08:39 trial_2e2a203129c19a297e3d27217a5f032e
# drwxr-xr-x  3 mustafamuratarat  staff    96 Mar  1 08:39 trial_5c3d1edad99aa9d7d83d12ec027e9c6d
# drwxr-xr-x  3 mustafamuratarat  staff    96 Mar  1 08:39 trial_68c2af8a6ca1f602ba37ca1eec9c23a9
```

Her model hakkında bilgi almak için:

```python
!ls ../output/tuner-1
#1 2 3 4 5 6
```

Her model için, `tuner-1` dizini model mimarisi, farklı kontrol noktaları, değerlendirme verileri ve bunun gibi bir çok bilgiyi içerir. Birinci modelin mimarisini okumaya bir örnek aşağıda gösterilmiştir:

```python
!ls ../output/tuner-1/1
# 1.arch.txt                        model.ckpt-0.meta
# checkpoint                        model.ckpt-10.data-00000-of-00001
# eval                              model.ckpt-10.index
# graph.pbtxt                       model.ckpt-10.meta
# model.ckpt-0.data-00000-of-00001  replay_config.pbtxt
# model.ckpt-0.index
```

ve aşağıdaki komut model 1'in grafiğini verecektir. Bu grafik dosyasının çıktısı çok uzun olduğu için yazdırmıyorum. Onun yerine, TensorBoard kullanarak görselleştirebilirim.

```
! cat ../output/tuner-1/1/graph.pbtxt
```

### Tensorboard ile grafik nesnesini görüntüleme
Tensorboard kullanarak, bir `.pbtxt` formatlı dosyadan TensorFlow grafiğini görüntüleyebilirsiniz. Önce TensorBoard servisini başlatmalısınız.

1. Komut istemini (Windows) veya terminali (Ubuntu / Mac) açın
2. Projenizin olduğu ana dizine gidin.
3. Python'un `virtualenv` paketini kullanıyorsanız, TensorFlow'u kurduğunuz sanal ortamı etkinleştirin.
4. Python aracılığıyla TensorFlow kütüphanesini görebildiğinizden emin olun
  * Python3 yazın, `>>>` görünümlü bir bilgi istemi alacaksınız
  * `import tensorflow as tf` deneyin.
  * Yukaridaki kodu başarılı bir şekilde çalıştırabilirseniz, devam edebilirsiniz.
5. `exit()` yazarak Python isteminden (yani `>>>`) çıkın ve aşağıdaki komutu yazın:
  * `tensorboard --logdir=output`
  * `--logdir`, görselleştirmek istediğimiz AutoML modellerin çıktısının bulunduğu dizindir.
  * İsteğe bağlı olarak, TensorBoard'un çalıştığı bağlantı noktasını değiştirmek için `--port = <port_you_like>` kullanabilirsiniz.
6. Şimdi şu mesajı almalısınız: `TensorBoard 2.2.2 at http://localhost:6006/ (Press CTRL+C to quit)`.
7. `http://localhost:6006/`'yı web tarayıcısına girin. Artık `graph.pbtxt` dosyasını tüm modeller için görüntüleyebilirsiniz!

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202021-03-01%20at%2009.19.01.png?raw=true)

### Sanal ortamı devre dışı bırakmak ve silmek

Çalışmalarınız tamamlandıktan sonra yarattığınız sanal ortamı silmek için yapmanız gereken pek bir şey yoktur.

Sanal ortamı devre dışı bırakmak için `deactivate` komutunu çalıştırabilirsiniz. 

```bash
(venv) Arat-MacBook-Pro:tutorial1 mustafamuratarat$ deactivate
Arat-MacBook-Pro:tutorial1 mustafamuratarat$ 
```

Daha sonra uygulamanızı yinelemeli olarak (`rm -rf venv`) kaldırarak artık dosyalardan kurtulabilirsiniz. 

### Jupyter çekirdeğini kaldırmak

Sanal ortamınızı sildikten sonra, onu Jupyter'den çekirdek kaldırmak isteyeceksiniz. İlk olarak JupyterLab'de hangi çekirdeklerin mevcut olduğunu görelim. Bunları şu şekilde listeleyebilirsiniz:

```
Arat-MacBook-Pro:~ mustafamuratarat$ jupyter kernelspec list
Available kernels:
  venv       /Users/mustafamuratarat/Library/Jupyter/kernels/venv
  python3    /Library/Frameworks/Python.framework/Versions/3.8/share/jupyter/kernels/python3
```

Şimdi, çekirdeği kaldırmak için şunu yazabilirsiniz:

```
jupyter kernelspec uninstall venv
```
