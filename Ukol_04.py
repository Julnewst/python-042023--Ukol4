"""
Úkol 4
Stáhni si data o výsledcích marketingové kampaně jedné portugalské banky, které jsou v souboru ukol_04_data.csv. Data mají následující proměnné.

První skupina proměnných zahrnuje obecné informace o klientovi/klientce.

age = věk (číslo)
job = typ zaměstnání (kategorická proměnná, obsahuje následující možnostti: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student", "blue-collar","self-employed","retired","technician","services")
marital = rodinný stav (kategorická proměnná, obsahuje možnosti "married","divorced","single", "divorced" zahrnuje rozvedené i ovdovělé)
education = vzdělání (kategorická proměnná, obsahuje následující možnosti: "unknown","secondary","primary","tertiary")
default = má úvěr v prodlení (binární proměnná, obsahuje možnosti "yes","no")
balance = průměrný zůstatek na účtu (numerická proměnná, v eurech)
housing: má úvěr na bydlení (binární proměnná, obsahuje možnosti "yes", "no")
loan: má osobní půjčku (binární proměnná, zahrnuje možnosti "yes","no")
Druhá skupina proměnných se týká posledního kontaktu v aktuální kampani

contact = způsob navázání kontaktu (kategorická proměnná, obsahuje možnosti "unknown","telephone","cellular")
day = den v měsíci posledního kontaktu (číselná proměnná)
month = měsíc posledního kontaktu (kategoriální proměnná, obsahuje možnosti "jan", "feb", "mar", …, "nov", "dec")
duration = délka posledního kontaktu v sekundách (číselná proměnná)
Třetí skupina obsahuje zbývající vstupní proměnné

campaign = počet kontaktů během aktuální kampaně (včetně posledního, číselná proměnná)
pdays = počet dnů uplynulých od posledního kontaktu s klientem (číselná proměnná, obsahuje -1, pokud klient/klientka zatím nebyl(a) kontaktována)
previous = počet kontaktů před stávající kampaní (číselná proměnná)
poutcome = výsledek předchozí kampaně (kategorická proměnná, obsahuje možnosti "unknown","other","failure","success")
Výstupní proměnná

y = informace, zda si klient/klientka založil(a) termínovaný účet (binární proměnná, obsahuje možnosti ano/ne)
Pro splnění úkolu je třeba provést následující body.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    accuracy_score, 
    precision_score, 
    recall_score
    )
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import export_graphviz
from sklearn.svm import LinearSVC, SVC
from six import StringIO
from IPython.display import Image  
import pydotplus
from pydotplus import graph_from_dot_data
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

import os
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

#načtení dat
data = pd.read_csv('Ukol_04_data.csv')
print(data.head())

#Vytvoř rozhodovací strom na základě všech vstupních proměnných, s výjimkou proměnných day a month. Výstupní proměnnou je informace, zda si klient založní termínovaný účet. Omez výšku stromu na 4 patra a vygeneruj obrázek (v rámci Jupyter notebooku nebo jej ulož jako samostatný obrázek). Kategoriální proměnné uprav pomocí OneHotEncoder, číselné proměnné nijak upravovat nemusíš. Dále vytvoř matici záměn a urči výši metriky accuracy.

import numpy

#definice výstupní proměnné
y = data['y']

#rozdělení dat na kategoriální a numerická
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
numeric_columns = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous' ]

#převedení numerických dat na pole
numeric_data = data[numeric_columns].to_numpy()

#úprava kategoriálních hodnot převodem textových hodnot na číselné hodnoty pomocí binárního vektoru.

ohe = OneHotEncoder()
encoded_columns = ohe.fit_transform(data[categorical_columns])
encoded_columns = encoded_columns.toarray()

#sloučení převedených dat
X = numpy.concatenate([encoded_columns, numeric_data], axis=1)

#rozdělení dat na testovací a tréninková
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#vytvoření rozhodovacího stromu
clf = DecisionTreeClassifier(max_depth=4)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#vygenerování obrázku stromu v png
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, feature_names=list(ohe.get_feature_names_out()) + numeric_columns, class_names=["No", "Yes"])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('tree_1.png')

#matice záměn
ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
plt.show()
#Matice záměn - správně predikovaných "yes" je 538, nesprávně 1173, "no" správně 12917, nesprávně 292. 

#metrika pro strom - Vypočítej hodnotu vybrané metriky pro rozhodovací strom, který byl vytvořen v prvním bodě.


#opakujeme úpravu dat - bez omezení počtu úrovní, rozdělení na skupiny
scaler = StandardScaler() #normalizace dat
numeric_data = scaler.fit_transform(data[numeric_columns])

ohe = OneHotEncoder() #převedení kategoriálních dat na číselné hodnoty
encoded_columns = ohe.fit_transform(data[categorical_columns])
encoded_columns = encoded_columns.toarray()

X = numpy.concatenate([encoded_columns, numeric_data], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

"""Accuracy (přesnost) udává celkovou míru správnosti modelu, což zahrnuje správně predikované pozitivní i negativní případy. V případě nevyvážených datových sad, kdy je většina případů negativních (klienti nezainteresovaní o termínovaný účet), může být tato metrika klamná, protože model může dosáhnout vysoké accuracy pouze tím, že většinou predikuje negativní třídu."""

print(accuracy_score(y_test, y_pred))

# V tomto případě je celková míra správnosti modelu 90% (0.9018096514745308)

"""
Vedení marketingového oddělení banky rozhodlo, že chce využít strojové učení k efektivní kampani. Chce ale vybrat nejlepší algoritmus, který bude predikovat, kdo z klientů má o termínovaný účet zájem. Následně bude kontaktovat ty, u kterých model predikuje zájem, a nebude kontaktovat ty, u kterých model bude predikovat nezájem. Algoritmus bude vybrán na základě jedné z metrik, které jsme si ukazovali na 9. lekci. Vedení marketingového oddělení se chce vyhnout zbytečnému kontaktování klientů, kteří o termínovaný účet nemají zájem. Nevadí, pokud se neozvou někomu, kdo o termínovaný vklad zájem má. Vyber podle této preference vedení vhodnou metriku. Metriku napiš jako komentář v programu nebo jej doplň do buňky v Jupyter notebooku.
"""

print(precision_score(y_test, y_pred, pos_label='yes'))

#Precision (přesnost) je metrika, která udává podíl správně predikovaných pozitivních případů (tj. klienti, kteří mají zájem o termínovaný účet) mezi všemi případy, které byly klasifikovány jako pozitivní (tj. klienti, které model predikoval jako mající zájem). Vedení marketingového oddělení se zajímá o minimalizaci falešně pozitivních případů, tedy kontaktování klientů, kteří ve skutečnosti o termínovaný účet nejsou zainteresováni. V tomto případě je podíl správně identifikovaných pozitivních případů na všech identifikovaných pozitivních případech asi 65% (0.6481927710843374)


#Vypočítej hodnotu vybrané metriky pro rozhodovací strom, který byl vytvořen v prvním bodě. Nejsem si jista úkolem, protože accuracy je požadováno v 1. bodě a v dalším se počítá s precision. Tedy dávám 3. do počtu. 

print(recall_score(y_test, y_pred, pos_label='yes'))

#Recall je definován jako podíl počtu správně klasifikovaných pozitivních instancí (true positives) ku celkovému počtu pozitivních instancí (true positives + false negatives). Z celkového počtu skutečných pozitivních případů z testovacích dat, byl model schopen správně identifikovat pouze 31.44% z nich.

"""
Využij algoritmus K Nearest Neighbours k predikci, zda si klient/klientka založí termínovaný účet. Využij všechny vstupní proměnné, s výjimkou proměnných day a month. Kategoriální proměnné uprav pomocí OneHotEncoder (tj. stejně jako u rozhodovacího stormu). Na číselné proměnné tentokrát použij StandardScaler. Pomocí cyklu (nebo pomocí GridSearchCV) urči počet uvažovaných sousedů, které algoritmus bere v úvahu. Uvažuj následující hodnoty parametru: 3, 7, 11, 15, 19, 23. Jaká je nejlepší hodnota metriky? A je lepší než u rozhodovacího stromu?
"""

clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

model_1 = KNeighborsClassifier()
params_1 = {"n_neighbors": [3, 7, 11, 15, 19, 23]}

clf_1 = GridSearchCV(model_1, params_1, scoring="accuracy")
clf_1.fit(X, y)

print(clf_1.best_params_)
print(round(clf_1.best_score_, 2))

#Tedy náš model je nejpřesnější, pokud jako parametr použijeme hodnotu 23, pak dosahuje správnosti při klasifikaci vzorků 88%. Metrika rozhodovacího stromu je trochu přesnější - 90%.


model_2 = SVC(kernel="linear")
params_2 = {"decision_function_shape": ["ovo", "ovr"]}

clf_2 = GridSearchCV(model_2, params_2, scoring="accuracy")
clf_2.fit(X, y)

print(clf_2.best_params_)
print(round(clf_2.best_score_, 2))

"""
Výsledek:
{'decision_function_shape': 'ovo'}
0.89
"""
#Lineární verze Support Vector Machine s parametrem decision_function_shape nastaveným na "ovo" dosáhla vysoké přesnosti (89 %) při klasifikaci dat. Je tedy malinko přesnější než metoda K-nearest neighbors. Při velkém objemu dat je ale náročnější na jejich zpracování a trvá delší čas. Zmenšení vzorku by mohla zase znamenat jeho zkreslení.

#Porovnej hodnoty metrik pro rozhodovací strom, K Nearest Neighbours a Support Vector Machine. Ve kterém z bodů jsme dosáhli nejvyšší hodnoty metriky?

"""Výsledky:
Accuracy strom: 0.9018096514745308
Accuracy  K Nearest Neighbours: {'n_neighbors': 23} 0.88
Accuracy Support Vector Machine: {'decision_function_shape': 'ovo'} 0.89

Strom omezený na čtyři patra je nepřesnější.
"""
#Bonus: Pomocí cyklu vyzkoušej další možné výšky rozhodovacího stromu, například v rozmezí 5 až 12.

model_3 = DecisionTreeClassifier()

# Definování hodnot výšek rozhodovacího stromu
depths = list(range(5, 13)) #převod na seznam
accuracy_scores = [] # seznam hodnot accuracy score

for depth in depths:
    params = {"max_depth": [depth]}
    clf = GridSearchCV(model_3, params, scoring="accuracy")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))
plt.plot(depths, accuracy_scores)
plt.title('Max depth vs Accuracy score') #Název grafu
plt.grid(True)  # Přidání mřížky
plt.xlabel("Max Depth")  # Popis osy x
plt.ylabel("Accuracy Score")  # Popis osy y
plt.show() #zobrazí graf na obrazovce
print(accuracy_scores)
    
# Při hloubce rozhodovacího stromu rovné 8 dosahuje model nejlepších výsledků ve srovnání s ostatními testovanými hloubkami. Hodnota přesnosti je 0.9020777479892761.
    