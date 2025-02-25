
Vi arbejder i Google Colab men andre værktøjer kan bruges. Fx Jupiter Notebook, IDLe eller VSCode. 
I https://colab.research.google.com/ vælges fil ->Ny  notebook i Drev


---

## **Klassifikation med Iris-datasættet**


### **Opdag**
Vi vil undersøge hvad datasættet indeholder. Indsæt kode i et kodefelt i Colab.

Kode 1:
```python
from sklearn.datasets import load_iris
import pandas as pd

# Indlæs datasættet
iris = load_iris()

# Konverter til en pandas DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df['target_name'] = iris_df['target'].apply(lambda x: iris.target_names[x])

# Vis de første rækker i datasættet
print(iris_df.head())
```

**Læg mærke til**:
- Datasættet indeholder data om tre typer iris-blomster: Setosa, Versicolor og Virginica.
- Egenskaber: `sepal length`, `sepal width`, `petal length`, `petal width`.
- Mål: Klassificere blomsten ud fra disse egenskaber.

---

### **Opdag: Visualiser**
Vi vil undersøge data ved hjælp af plots. Indsæt koden i et nyt kodefelt

Kode 2:
```python
import matplotlib.pyplot as plt

# Scatterplot for to funktioner
plt.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target, cmap='viridis')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Scatterplot of Iris Data')
plt.show()
```


- Brug forskellige par af egenskaber for at undersøge data.
- Hvordan ser de tre klasser ud i plottet?

---

### **Split**
Vi vil opdele data i trænings- og testdata. Indsæt koden i et nyt kodefelt.

Kode 3:
```python
from sklearn.model_selection import train_test_split

# Opdel data
# Stratificeret opdeling
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
)



print(f"Antal træningsdata: {len(X_train)}")
print(f"Antal testdata: {len(X_test)}")

```


- Hvorfor deler vi data?
- Hvad betyder `test_size`?

---
### **Vælg en model og træn (Logistisk Regression)**
Vi vil træne data i en simpel klassifikationsmodel (Logistisk Regression). Indsæt koden i et nyt kodefelt.

Kode 4:
```python
from sklearn.linear_model import LogisticRegression

# Træn modellen
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Forudsig med testdata
y_pred = model.predict(X_test)

print("Forudsigelser:", y_pred)
```


- Hvad betyder "træning" af en model?
- Hvilke data laver vi forudsigelser på?
  

---

### **Evaluer (Logistisk Regression)**
Vi vi måle modellens nøjagtighed. Indsæt koden i et nyt kodefelt.

Kode 5:
```python
from sklearn.metrics import accuracy_score, classification_report

# Evaluer nøjagtighed
accuracy = accuracy_score(y_test, y_pred)
print(f"Modelens nøjagtighed: {accuracy:.2f}")


```


- Hvad fortæller nøjagtighed?

---

### **Evlauer: Visualiser (Logistisk Regression)**
Vi vil lave en confusionsmatrix. Indsæt koden i et ny kodefelt.

Kode 6:
```python
import numpy as np

# Lav en confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=iris.target_names, cmap='viridis')
plt.title("Confusion Matrix")
plt.show()
```

- Hvad viser en confusion matrix?
- Hvor ser modellen ud til at fejle?

---

### **Vælg en anden model og træn (KNN)**
Vi vil eksperimentére med en anden model, K-Nearest Neighbors (KNN).
Nå vi træner en anden model skal vi importere den nye model og ændre den linje kode, hvor vi vælger modellen.
For at beholde den gamle kørsel kan den gamle kode kopieres i et nyt kodefelt.
Vi starter med at indsætte kode 4 fra afsnittet **Vælg model og træn**

```python
# Træn modellen
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Forudsig med testdata
y_pred = model.predict(X_test)
print("Forudsigelser:", y_pred)
```

Nu skal vi tildele den nye model til variablen model og for at gøre det, er vi nød til at importerer det rigtige bibliotek.
```python
from sklearn.neighbors import KNeighborsClassifier


model = KNeighborsClassifier(n_neighbors=3)
```

Indsæt linjen med import og ændr linjen med model og træn.

### **Evaluer (KNN)**
Her kan vi bruge koden uden ændringer fra afsnittet Evaluer. Indsæt koden fra kode 5.

Evaluer modellen og lav confusion matrix med samme kode som før


### **Tune (KNN)**
Vi vil prøve med forskellig antal naboer,



- Vælg et nyt tal for antal naboer (`n_neighbors`)

### **Evaluer (KNN)**
Vi skal evaluere hvergang vi har valgt et andet antal naboer.


---



---

### **Vælg ny model og træn (Klassifikationstræ)**
Vi kan endnu engang genbruge koden fra kode 4. Derefter skal vi importere det rigtige bibliotek og ændre variablen model.

```python
from sklearn.tree import DecisionTreeClassifier

# Initialiser modellen
model = DecisionTreeClassifier(max_depth=3, random_state=42)

```

---

### **Evaluer (Klassifikationstræ)**
Evaluer modellen og lav confusion matrix med samme kode som før

---

### **Visualiser klassifikationstræet**
Vi kan visualisere klassifikationstræet
```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Visualiser træet
plt.figure(figsize=(12, 8))
plot_tree(tree_model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Klassifikationstræ")
plt.show()
```

---

### **Tune og evaluer (Klassifikationstræ)**

- Prøv med forskellige `max_depth` ?
- Overvej hvordan  hvilke egenskaber der opdeles efter først?
- Er modellen overfittet eller underfittet? Hvordan kan du ændre dette?

---

### **Tune mere (Klassifikationstræ)**
- Eksperimentér med andre parametre, fx:
  - `criterion="gini"` eller `criterion="entropy"`.
  - `min_samples_split` for at begrænse, hvor mange data punkter en node skal have før split.
- Sammenlign præstationen med de andre modeller som Logistisk regression og K-Nearest Neighbors.

---
