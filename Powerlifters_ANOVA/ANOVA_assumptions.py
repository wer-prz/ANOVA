import pandas as pd
from scipy.stats import shapiro
from scipy.stats import levene
import seaborn as sns
from scipy.stats import kstest, norm
from scipy.stats import zscore
import matplotlib.pyplot as plt
from scipy.stats import anderson

df = pd.read_excel('Powerlifters.xlsx', usecols="E,I")

df_fil = df[(df['STBR'] >= 0.7) & (df['STBR'] <= 1.9)].copy()

#Podział grup względem wieku
g1=26
g2=31

def przypisz_grupe_wiekowa(wiek):
    if wiek <= g1:
        return 'Młodsi'
    elif wiek <= g2:
        return 'Średni'
    else:
        return 'Starsi'

df_fil['GrupaWiekowa'] = df_fil['Age'].apply(przypisz_grupe_wiekowa)

df1=df_fil[df_fil['GrupaWiekowa']=="Młodsi"]
df2=df_fil[df_fil['GrupaWiekowa']=="Średni"]
df3=df_fil[df_fil['GrupaWiekowa']=="Starsi"]

print(f"Młodsi: {len(df1)}")
print(f"Średni: {len(df2)}")
print(f"Starsi: {len(df3)}")

#---TESTY NORMALNOŚCI

#Kolmogorov Smirnov
print("\n")
print("Test Kolmogorova Smirnova: ")
for grupa, dane in zip(['Młodsi', 'Średni', 'Starsi'], [df1, df2, df3]):
    wartosci = dane['STBR']
    wartosci_z = zscore(wartosci)
    stat, p_val = kstest(wartosci_z, 'norm')
    print(f"{grupa}:")
    print(f"  Statystyka KS: {round(stat, 4)}")
    print(f"  p-value:     {round(p_val, 4)}")
    if p_val >=0.05:
        print("p-value większe niż 0.05 - Brak podstaw do odrzucenia normalności\n")
    else:
        print("p-value mniejsze niż 0.05 - Odrzucamy normalność\n")

#Shapiro-Wilk
print("\n")
print("Test Shapiro-Wilka: ")
for grupa, dane in zip(['Młodsi', 'Średni', 'Starsi'], [df1, df2, df3]):

    wartosci = dane['STBR']  # wybieramy kolumnę z wartością przysiad / klatka
    stat, p_val = shapiro(wartosci)
    print(f"{grupa}:")
    print(f"  Statystyka:     {round(stat, 4)}")
    print(f"  p-wartość:      {round(p_val, 4)}\n")
    if p_val >=0.05:
        print("p-value większe niż 0.05 - Brak podstaw do odrzucenia normalności\n")
    else:
        print("p-value mniejsze niż 0.05 - Odrzucamy normalność\n")

# Lista grup i danych

for nazwa, dane in zip(['Młodsi', 'Średni', 'Starsi'], [df1, df2, df3]):

    wartosci=dane['STBR']
    wynik = anderson(wartosci, dist='norm')
    stat = wynik.statistic
    granica = wynik.critical_values[2]
    print(f"\n{nazwa}: stat = {round(stat, 4)}, granica = {round(granica, 4)}")
    if stat > granica:
        print("p-value mniejsze niż 0.05 - Odrzucamy normalność\n")
    else:
        print("p-value większe niż 0.05 - Brak podstaw do odrzucenia normalności\n")

#---WYKRESY---

plt.figure(figsize=(15, 4))

for i, (nazwa, dane) in enumerate(zip(['Młodsi', 'Średni', 'Starsi'], [df1, df2, df3])):
    wartosci=dane["STBR"]
    plt.subplot(1, 3, i + 1)
    sns.histplot(wartosci, kde=True, bins=30, color='blue')
    plt.title(nazwa)
    plt.xlabel('STBR (Squat / Bench)')
    plt.ylabel('Liczba zawodników')

plt.tight_layout()
plt.show()