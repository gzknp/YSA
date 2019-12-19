"""
OĞUZ KAAN PARLAK
120101047
"""
#Kütüphanelerin tanımlanması
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Dosyadan veriyi okuma ve giriş-çıkış sütunlarını ayarlama
dataset = pd.read_csv('C:/Users/kaan/Desktop/Yeni klasör/heart.csv')
Giris= dataset.iloc[:,0:13]
Cikis= dataset.iloc[:,13]
from sklearn.preprocessing import StandardScaler
olcek = StandardScaler()
Giris = olcek.fit_transform(Giris)
#Veri setini Train-Test olarak ayırma
from sklearn.model_selection import train_test_split
Giris_train, Giris_test, Cikis_train, Cikis_test = train_test_split(Giris, Cikis, test_size=0.15)
#YSA mimarisini oluşturma
from keras import Sequential
from keras.layers import Dense
sinif = Sequential()
#Gizli katman
sinif.add(Dense(16, activation='relu', input_dim=13))
#Çıkış katmanı
sinif.add(Dense(1, activation='sigmoid'))
#Optimizer belirlenmesi(learning rate gibi parametrelerin belirlenmesi)
ada=keras.optimizers.SGD(learning_rate=0.001,momentum=0.9)
#Sistemin belirlenen optimizer ve belirttiğimiz loss function'a göre compile edilmesi 'mean_squared_error' şeklinde loss fonksiyonu değiştirilebilir.
sinif.compile(optimizer =ada,loss='binary_crossentropy', metrics =['acc'])
#Sistemin eğitilmesi--validation split train verisi içinde ayrılacak doğrulama verisini ifade eder.
#Epoch periyot sayısını belirtmek için kullanılır.
history=sinif.fit(Giris_train,Cikis_train, validation_split=0.175,batch_size=10, epochs=60,verbose=2)
#Sistemin test verileri üzerinden test edilmesi
kayip_orani, dogruluk_orani =sinif.evaluate(Giris_test,Cikis_test,verbose=1,batch_size=10)
print(" ")
print("--------Eğitilen sisteme sokulan Test verisinin accuracy ve loss oranları----------")
print(" ")
print('test loss oranı ',kayip_orani)
print('test accuracy oranı',dogruluk_orani)
print(" ")
#Sistemin test verisi üzerinden tahminde bulunarak sınıflandırma yapması
Cikis_tahmin=sinif.predict_proba(Giris_test)
#print(Cikis_tahmin)
Cikis_tahmin=(Cikis_tahmin>0.5)#Tahmin edilen çıkış 0.5'in üzerinde ise sonucu 1(hasta), altında ise 0(sağlıklı) olarak kabul et.
#Roc curve plot fonksiyonu
from sklearn.metrics import roc_curve, auc
def plot_roc(tahmin,Cikis):
    fpr,tpr, _=roc_curve(Cikis, tahmin)
    roc_auc=auc(fpr,tpr)
    plt.figure()
    plt.plot(fpr,tpr,label='ROC curve (area=%0.2f)'%roc_auc)
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
plot_roc(Cikis_tahmin, Cikis_test)
from sklearn.metrics import confusion_matrix
import itertools
cm = confusion_matrix(Cikis_test, Cikis_tahmin)
#confusion matrix plot fonksiyonu
def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion Matrix',cmap=plt.cm.Blues):
    
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    print('Confusion matrix')
    print(cm)
    thresh=cm.max()/2.
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],
                horizontalalignment="center",
                color="brown" if cm[i,j]>thresh else "brown")
    plt.tight_layout()
    plt.ylabel('Gerçek değerler')
    plt.xlabel('Tahmin değerleri')
cm_plot_labels = ['Sağlıklı','Hasta']
plot_confusion_matrix(cm,cm_plot_labels,title='Confusion Matrix')
plt.show()
#test verileriyle beklenen çıkışların karşılaştırılldığı confusion matrixten gelen verilerle oluşturulan sınıflandırma raporu
from sklearn.metrics import classification_report
print(classification_report(Cikis_test, Cikis_tahmin))
#Train--Validation loss değerinin epocha göre değişim grafiğinin gösterimi
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
#Train--Validation accuracy değerinin epocha göre değişim grafiğinin gösterimi
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
