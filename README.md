# Voice Recognition – Gender & Age Classification  
Projekt u Pythonu za prepoznavanje spola i dobne skupine govornika korištenjem RF i DNN modela

## Opis projekta  
Ovaj projekt razvijen je u suradnji s još dva kolege, s ciljem izgradnje sustava za klasifikaciju govornika na temelju kratkih audio isječaka. Sustav prepoznaje:

- **Spol govornika:** muško / žensko  
- **Dobnu skupinu:** dijete / odrasla osoba  

Za izgradnju klasifikatora korištene su dvije različite metode:  
- **Random Forest (RF)**  
- **Deep Neural Network (DNN)**  

Projekt obuhvaća cijeli proces – od preuzimanja i pripreme podataka do treniranja modela te generiranja rezultata i matrica zabune.

---

## Dataset  
Korišten je **Mozilla Common Voice** dataset, iz kojeg su preuzeti relevantni audio uzorci.

### Obrada podataka uključivala je:  
- **Downsampling** audio zapisa na uniformnu frekvenciju  
- **Normalizaciju i čišćenje** signala  
- **Podjelu dataset-a** po klasama kako bi distribucija bila uravnotežena:  
  - 50% muški  
  - 50% ženski  
  - 50% djeca  
  - 50% odrasli  

- Generiranje **MFCC značajki** (uz delte i druge dodatne značajke kad je potrebno)

---

##  Metode i modeli  
### Značajke (Feature Extraction)
 - 13 glavninh frekvencijskih značajki govora kod MFCC-a
 
### 1. Random Forest  

### 2. Deep Neural Network (DNN)  

---
