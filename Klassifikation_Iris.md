
Vi arbejder i Google Colab men andre værktøjer kan bruges. Fx Jupiter Notebook, IDLe eller VSCode. 
I https://colab.research.google.com/ vælges fil ->Ny  notebook i Drev


---

## **Klassifikation med Iris-datasættet**


### **Opdag**
Vi vil undersøge hvad datasættet indeholder.

Kode:
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
Vi vil undersøge data ved hjælp af plots.

Kode:
```python
import matplotlib.pyplot as plt

# Scatterplot for to funktioner
plt.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target, cmap='viridis')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Scatterplot of Iris Data')
plt.show()
```

**Opgave**:
- Brug forskellige par af egenskaber for at undersøge data.
- Hvordan ser de tre klasser ud i plottet?

---

### **Split**
Vi vil opdele data i trænings- og testdata

Kode:
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

**Opgave**:
- Hvorfor deler vi data?
- Hvad betyder `test_size`?

---

### **Træn**
Vi vil træne data i en simpel klassifikationsmodel (Logistisk Regression).

Kode:
```python
from sklearn.linear_model import LogisticRegression

# Træn modellen
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Forudsig med testdata
y_pred = model.predict(X_test)

print("Forudsigelser:", y_pred)
```

**Opgave**:
- Hvad betyder "træning" af en model?
  

---

### **Evaluer**
Vi vi måle modellens nøjagtighed.

Kode:
```python
from sklearn.metrics import accuracy_score, classification_report

# Evaluer nøjagtighed
accuracy = accuracy_score(y_test, y_pred)
print(f"Modelens nøjagtighed: {accuracy:.2f}")

# Klassifikationsrapport
print("Klassifikationsrapport:")
```

**Opgave**:
- Hvad fortæller nøjagtighed?

---

### **Evlauer (visualiser)**
Vi vil lave en confusionsmatrix.

Kode:
```python
import numpy as np

# Lav en confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=iris.target_names, cmap='viridis')
plt.title("Confusion Matrix")
plt.show()
```

**Opgave**:
- Hvad viser en confusion matrix?
- Hvor ser modellen ud til at fejle?

---

### **Vælg en anden model**
Vi vil eksperimentére med en anden model, K-Nearest Neighbors (KNN).

Kode:
```python
from sklearn.neighbors import KNeighborsClassifier

### **Træn**
# Træn KNN-modellen
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

### **Evaluer**
# Forudsig og evaluer
y_pred_knn = knn_model.predict(X_test)
print(f"KNN-modelens nøjagtighed: {accuracy_score(y_test, y_pred_knn):.2f}")
```


### **Tune**
**Opgave**:

- Hvordan kan vi vælge `n_neighbors`?
- Lav en konfusionsmatrix for knn-modellen.

---

Her er et simpelt eksempel på **supervised learning** og **klassifikation** ved hjælp af en **Decision Tree Classifier** (klassifikationstræ). Datasættet, vi bruger, er stadig **Iris-datasæt**.

---

## **Undervisningsforløb: Klassifikation med Klassifikationstræer**
### **Mål**
1. Introducere begrebet klassifikationstræer.
2. Forstå, hvordan beslutningstræer bruges i supervised learning.
3. Træne, evaluere og visualisere et beslutningstræ ved hjælp af Iris-datasættet.

---

### **Baggrundsviden: Hvad er et klassifikationstræ?**
- Et **klassifikationstræ** opdeler data i undergrupper baseret på betingelser.
- Træet består af **noder**:
  - **Rodnode**: Startpunktet, hvor data opdeles første gang.
  - **Indre noder**: Repræsenterer yderligere opdelinger.
  - **Blade**: Slutnoderne, som indeholder en klassifikation (fx Setosa, Versicolor, Virginica).

Træet opbygger opdelinger ved at vælge funktioner (features) og værdier, der bedst adskiller klasserne.

---

### **Trin 1: Forstå Iris-datasættet**
**Formål**: Lær, hvad datasættet indeholder.

Iris-datasættet har:
- 4 funktioner (features): `sepal length`, `sepal width`, `petal length`, `petal width`.
- 3 klasser: `Setosa`, `Versicolor`, `Virginica`.

---

### **Trin 2: Visualisering af data**
Visualisering gør det lettere at forstå, hvordan funktionerne skelner mellem klasser.

**Aktivitet**:
1. Brug scatterplots til at se datafordelingen.
2. Udforsk, hvilke funktioner der bedst adskiller klasserne.

---

### **Trin 3: Opdel datasættet**
Supervised learning kræver, at data opdeles i:
- **Træningsdata**: Bruges til at træne modellen.
- **Testdata**: Bruges til at evaluere modellens præcision.

**Aktivitet**:
1. Brug `train_test_split` til at dele data.
2. Sørg for, at hver klasse er repræsenteret i både trænings- og testdata.

---

### **Trin 4: Træning af klassifikationstræet**
**Formål**: Træn et klassifikationstræ, så det lærer at skelne mellem klasser.

**Aktivitet**:
- Brug en Decision Tree Classifier til at træne modellen.
- Eksperimentér med træets parametre, som fx `max_depth` (begrænsning af træets dybde).

---

### **Trin 5: Evaluering af modellen**
Evaluering er vigtig for at sikre, at modellen generaliserer godt.

**Aktivitet**:
- Brug nøjagtighed som en simpel præstationsmåling.
- Lav en confusion matrix for at se, hvor modellen klassificerer korrekt eller fejler.

---

### **Trin 6: Visualisering af beslutningstræet**
Beslutningstræer kan visualiseres for at gøre dem mere intuitive.

**Aktivitet**:
- Tegn træet og analyser, hvordan det træffer beslutninger baseret på funktionerne.
- Diskutér, hvilke funktioner træet finder mest informative.

---

## **Eksempler på Kode**

### **1. Indlæs Iris-datasættet**
```python
from sklearn.datasets import load_iris
import pandas as pd

# Indlæs datasættet
iris = load_iris()

# Konverter til pandas DataFrame for bedre visning
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df['target_name'] = iris_df['target'].apply(lambda x: iris.target_names[x])

print(iris_df.head())
```

---

### **2. Opdel data i træning og test**
```python
from sklearn.model_selection import train_test_split

# Opdel data
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

print(f"Træningsdata: {len(X_train)}, Testdata: {len(X_test)}")
```

---

### **3. Træn klassifikationstræet**
```python
from sklearn.tree import DecisionTreeClassifier

# Initialiser og træn modellen
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)
```

---

### **4. Evaluer modellen**
```python
from sklearn.metrics import accuracy_score, classification_report

# Forudsig testdata
y_pred = tree_model.predict(X_test)

# Evaluer
print(f"Nøjagtighed: {accuracy_score(y_test, y_pred):.2f}")
print("Klassifikationsrapport:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

---

### **5. Visualiser beslutningstræet**
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

## **Opgaver**
1. Hvordan påvirker `max_depth` modellens præstation?
2. Overvej hvordan træet vælger, hvilke funktioner det opdeler først?
3. Er modellen overfittet eller underfittet? Hvordan kan du ændre dette?

---

## **Udvidelse**
- Eksperimentér med andre parametre, fx:
  - `criterion="gini"` eller `criterion="entropy"`.
  - `min_samples_split` for at begrænse, hvor mange data punkter en node skal have før split.
- Sammenlign præstationen med andre modeller som Logistisk regression og K-Nearest Neighbors.

---
