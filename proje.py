import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.datasets import make_moons , make_circles , make_classification

from sklearn.neighbors import KNeighborsClassifier # klasik knn
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC # support vector 
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier , VotingClassifier # 3 farklı ağaç dal sınıflandırma ve regresyon uygulaması yapıyopr

import warnings
warnings.filterwarnings('ignore')# uyarıları engellemek


# data sests
#kendimiz herhangi bir veri seti kullanmadan veri seti oluşturacağız
random_state =42
n_samples = 20000
n_features = 10
n_classes = 2
noise_moon = 0.2
noise_circle = 0.2
noise_class = 0.2 #verinin ne kadar gürültülü olduğu aşağıdaki parametrelerde flip_y olarak işaretlicez
"""n_redundant anlamsız sample"""
""" n_clusters_per_class=1 olmasının sebebi 2^n alıyorm defaultu 2 olduğu için 2 üzeri 2 de 4 eder bu kadar features yok zaten ,,  n_class(2)*n_clustered_classses(2) çarpıldığında 4 çıkıyor bu da aslında bizde 2 adet özellik var)"""
#veri seti yapma yöntem 1
X,y = make_classification(n_samples = n_samples,
                    n_features=n_features,
                    n_classes=n_classes,
                    n_repeated=0,
                    n_redundant=0,
                    n_informative=n_features-1,
                    random_state=random_state,
                    n_clusters_per_class=1,
                    flip_y=noise_class)


data = pd.DataFrame(X)
data["target"] = y
plt.figure()
sns.scatterplot(x=data.iloc[:,0] , y =data.iloc[:,1] , hue="target" , data =data)

data_classification =(X,y)
#veri seti yapma yöntem 2 moon
moon = make_moons(n_samples = n_samples , noise=noise_moon , random_state=random_state)
data = pd.DataFrame(moon[0])
data["target"] = moon[1]
plt.figure()
sns.scatterplot(x=data.iloc[:,0] , y =data.iloc[:,1] , hue="target" , data =data)


#veri seti yapma yöntem 3 circle
circle = make_circles(n_samples = n_samples,  factor=0.1 , noise=noise_circle ,  random_state=random_state)
data = pd.DataFrame(circle[0])
data["target"] = circle[1]
plt.figure()
sns.scatterplot(x=data.iloc[:,0] , y =data.iloc[:,1] , hue="target" , data =data)

datasets = [moon,circle] # iki veri setide binary veri tipine benzer olduğundan birleştirebiliyoruz
n_estimators = 10


svc = SVC()
knn = KNeighborsClassifier(n_neighbors=15) # yüksek verme sebebimiz problemi çok iyi çözüyor
dt = DecisionTreeClassifier(random_state= random_state)
""" burda yaptığımız gibi istediğimiz algoritmayı koyup parametrelerini girip tanımlayıp , names ve classiferis kısmına tanımladığımızı eklersek çalışır"""
rf = RandomForestClassifier(n_estimators = n_estimators, random_state = random_state, max_depth = 2)
"""n_estimators = ağaç sayısı max_depth = derinlere inmesi olayı 2 verdiğimizdem derinlere inmesin çok demek """
ada = AdaBoostClassifier(n_estimators = n_estimators , random_state= random_state )

"""voting classifier algoritmaları(knn svm ada logistikregresyon noralnetwork..vss) teker teker deneyerek çoğunluğa karar verir . hard voting classifier çoğunluğu seçer , 
soft ise yüzdesel vererek bakar knn svm gibi alg larda yüzdesel yok bu yüzden hard kullanıcağız """

v1 = VotingClassifier(estimators=[('svc', svc),('knn', knn),('dt', dt),('rf', rf) , ('ada', ada)])



names = ["SVC" , "KNN" , "Decision Tree" , "Random Forest" ,"ADABOOST" , "Voting"]
classifiers =[svc , knn ,dt , rf ,ada ,v1]
"""gerekli algoritmaları oluşturduk"""

"""iki adet for bloğu olucak 1.blokta verisetlerini döndüreceğiz , 2.sinde algoritmaları çevireceğiz yani her algoritma 3 adet sınıflandırma yapacak"""
h = 0.2 # belirli çözünürlüğe bölmek için tanımladık 
i=1
figure = plt.figure(figsize=(18, 6))

for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = RobustScaler().fit_transform(X) # farklı olarak robus scaler i deniyoruz
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=random_state)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5 #x_min ve x_max bulunuyor
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5#y_min ve y_max bulunuyor
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),# bunları sonra belirli bir çözünürlükte (h) mashlere bölüyoruz
                         np.arange(y_min, y_max, h))
    
    #görselleştirme
    cm = plt.cm.RdBu # arka plan için
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    
    ax= plt.subplot(len(datasets) , len(classifiers)+1 , i)# +1 denmesinin sebeb, tabloda değişkenleri koymak
    if ds_cnt == 0:
     ax.set_title("Input data")
       
     ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,edgecolors='k')
     ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,marker = '^', edgecolors='k')
       
     ax.set_xticks(())
     ax.set_yticks(())
     i += 1
    
     print("Dataset # {}".format(ds_cnt))
     
     for name , clf in zip(names , classifiers): # CLF BİZİM ALGORİTMALARIMIZI TUTUYOR her for döngüsünde birini döndürücek
         ax = plt.subplot(len(datasets), len(classifiers) + 1, i)# görselleştirme kısmı
       
         clf.fit(X_train, y_train) # ilk önce eğitim setinde eğittik
       
         score = clf.score(X_test, y_test)# eğitimi test veri setinde denedik
       
         print("{}: test set score: {} ".format(name, score))
       
         score_train = clf.score(X_train, y_train)  # tekrardan aynısını yani test veri setindeki scoru eğitim verisi üzerinde test ediyoruz böylece test ve train sonuçlarını göreceğiz
         
       
         print("{}: train set score: {} ".format(name, score_train))
         print()
         # görselleştirme
         if hasattr(clf, "decision_function"): # burası decision tree için görselleştirmesi farklı o yüzden ona ayrı , diğerlerine ayrı görselleştirme yapılıcak
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
         else:
             Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
             
             
         Z = Z.reshape(xx.shape)
         ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

         # Plot the training points
         ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                        edgecolors='k')
         # Plot the testing points
         ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,marker = '^',
                       edgecolors='white', alpha=0.6)
            
         ax.set_xticks(())
         ax.set_yticks(())
         if ds_cnt == 0:
             ax.set_title(name)
         score = score*100
         ax.text(xx.max() - .3, yy.min() + .3, ('%.1f' % score),
                size=15, horizontalalignment='right')
         i += 1
         print("-------------------------------------")
         
plt.tight_layout()
plt.show()
def make_classify(dc, clf, name):
    x, y = dc
    x = RobustScaler().fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.4, random_state=random_state)
    
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print("{}: test set score: {} ".format(name, score))
        score_train = clf.score(X_train, y_train)  
        print("{}: train set score: {} ".format(name, score_train))
        print()

print("Dataset # 2")   
make_classify(data_classification, classifiers,names)           
                         
                 
       
       ##""" dikkatli incelersek burada fetures artığında svm daha iyi sonuç veriyor noise arttığında svm düşüyor bunları deneyerek görebiliriz bu projede"""
      
             ##    """svm büyük verilerde başarısız , birde anlaşırlığı zor , eğitimi zor ,yorum yapabilirliği zor """      
    ## knn basit ve güçlü algoritme , veri boyutu artığında öğrenme süresi artıyor ,  ayrıyaten aykırı değerlerden etkilenen bir algoritma
    ## knn da sınıf artığında svm e göre daha iyi sonuçlar elde edecektir
    # random forest anlaşılması zordur karışıktır , en önemli features leri bulma açısından  
    # ada boost algoritması design tree gibi ama zayıf dalı yani bireyi bularak iyileştiriyor ve ileri adımda bunu yaparak devam ediyor
    # voting classifier algoritmaları(knn svm ada logistikregresyon noralnetwork..vss) teker teker deneyerek çoğunluğa karar verir . hard voting classifier çoğunluğu seçer , 
    #soft ise yüzdesel vererek bakar knn svm gibi alg larda yüzdesel yok bu yüzden hard kullanıcağız """ 
    # bu yüzden voting classifer diğerlerine göre daha iyi sonuçlar döndürebilir
    # farklı algoritmaları birleştirerek daha iyi sonuç yapabiliriz
    #forestlar overthinki önlemede başarılıdır , adaboost da biasi azaltmakta etkili 
    #bias =modelin gerçeğe ne kadar uzak olduğunu gösterir
    #düşük bias = çok iyi uyum ve öğrenim fakat overfitting riski vardır
    # yüksek bias : model veriye fazla basitleştirilmişş yaklaşır underfitting yapar
    
    
    
    
    
    
    
    
    
    