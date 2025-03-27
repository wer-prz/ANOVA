from statsmodels.stats.power import FTestAnovaPower
from math import ceil
import pandas as pd
from scipy import  stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# parametry
alpha = 0.05  # poziom istotności (prawdopodobieństwo popełnienia błędu 1 rodzaju)
power = 0.8   # moc testu = 1 - beta, (beta - prawdopodobieństwo popełnienia błędu 2 rodzaju)
groups = 3    # liczba grup
# wskaźnik siły efektu f Cohena
fCohen = 0.25   # niski = 0.10, średni = 0.25, wysoki = 0.40

def group_size(alpha, power, groups, fCohen):
    return ceil(FTestAnovaPower().solve_power(effect_size=fCohen, k_groups=groups, alpha=alpha, power=power))

print("Alfa:", alpha, "\nMoc testu:", power, "\nLiczba grup:", groups, "\nWskaźnik f Cohena:", fCohen,
      "\nLiczebność każdej grupy:", group_size(alpha, power, groups, fCohen),
      "\nŁączna liczba badanych:", groups * group_size(alpha, power, groups, fCohen))

dane = pd.read_excel('dane.xlsx')  # wczytanie pliku

dane = dane.iloc[:, 1:]                 # usunięcie pierwszej kolumny

dane.columns = ['wiek', 'płeć', 'akt_fiz', 'spot_tow', 'ekran', 'sen', 'choroby', 'szczescie']
                                        # zmiana nazw kolumn

dane = dane[dane['wiek'] <= 35]         # usunięcie wierszy, w których wiek > 35

dane = dane.reset_index(drop=True)      # resetowanie indeksów

# funkcja do konwersji wartosci w kolumnach
def konwertuj(wartosc):
    if 'mniej niż' in wartosc:
        return 0
    elif 'więcej niż' in wartosc:
        return 9
    elif '-' in wartosc:
        return int(wartosc.split('-')[0].strip())
    elif 'Kobieta' in wartosc:
        return 'K'
    elif 'Mężczyzna' in wartosc:
        return 'M'
    elif 'Inne' in wartosc:
        return 'I'

dane['płeć'] = dane['płeć'].astype(str).apply(konwertuj)
dane['akt_fiz'] = dane['akt_fiz'].astype(str).apply(konwertuj)
dane['spot_tow'] = dane['spot_tow'].astype(str).apply(konwertuj)
dane['ekran'] = dane['ekran'].astype(str).apply(konwertuj)
dane['sen'] = dane['sen'].astype(str).apply(konwertuj)

# definiowanie przedziałów
bins = [-1, 1, 4, 11]   # przedziały
labels = [1, 2, 3]      # wartości w przedziałach


dane['grupa'] = pd.cut(dane['akt_fiz'], bins=bins, labels=labels, right=True) # tworzenie nowej kolumny 'grupa'

print(dane.head())
print(dane[['wiek', 'płeć', 'akt_fiz', 'spot_tow', 'ekran', 'sen', 'choroby', 'szczescie', 'grupa']].describe(include='all'))
                        # opis statystyczny kolumn


# zliczanie wartości w przedziałach

print("Zliczanie wartości w kolumnie 'grupa':")
print(dane['grupa'].value_counts())

# print("Zliczanie wartości w kolumnie 'wiek':")
# print(dane['wiek'].value_counts())
#
# print("\nZliczanie wartości w kolumnie 'płeć':")
# print(dane['płeć'].value_counts())
#
# print("\nZliczanie wartości w kolumnie 'akt_fiz':")
# print(dane['akt_fiz'].value_counts())
#
# print("\nZliczanie wartości w kolumnie 'spot_tow':")
# print(dane['spot_tow'].value_counts())
#
# print("Zliczanie wartości w kolumnie 'ekran':")
# print(dane['ekran'].value_counts())
#
# print("\nZliczanie wartości w kolumnie 'sen':")
# print(dane['sen'].value_counts())
#
# print("\nZliczanie wartości w kolumnie 'choroby':")
# print(dane['choroby'].value_counts())
#
# print("\nZliczanie wartości w kolumnie 'szczescie':")
# print(dane['szczescie'].value_counts())



zmienna = 'wiek'
# Podział na grupy
grupa_1 = dane[dane['grupa'] == 1][zmienna]
grupa_2 = dane[dane['grupa'] == 2][zmienna]
grupa_3 = dane[dane['grupa'] == 3][zmienna]
grupa_badana = grupa_1

# ----- TESTY NORMALNOŚCI -----
# test normalności Andersona-Darlinga
stat_anderson, krytyczne_wartosci, poziomy_istotnosci = stats.anderson(grupa_badana, dist='norm')
test_norm = 'Andersona-Darlinga'
print(f"Statystyka testu Andersona-Darlinga: {stat_anderson}")
print(f"Krytyczne wartości: {krytyczne_wartosci}")
print(f"Poziomy istotności: {poziomy_istotnosci}")
wartosc_krytyczna = krytyczne_wartosci[2]
if (stat_anderson > wartosc_krytyczna):
    print(f"Test normalności {test_norm}: Odrzucamy hipotezę o normalności rozkładu (poziom istotności: {alpha}).")
else:
    print(f"Test normalności {test_norm}: Brak podstaw do odrzucenia hipotezy o normalności (poziom istotności: {alpha}).")

# test normalności Shapiro-Wilka
stat_shapiro, p_shapiro = stats.shapiro(grupa_badana)
test_norm = 'Shapiro-Wilka'
print(f"\nStatystyka testu Shapiro-Wilka: {stat_shapiro}")
print(f"p-wartość: {p_shapiro}")
if (p_shapiro < alpha):
    print(f"Test normalności {test_norm}: Odrzucamy hipotezę o normalności rozkładu (poziom istotności: {alpha}).")
else:
    print(f"Test normalności {test_norm}: Brak podstaw do odrzucenia hipotezy o normalności (poziom istotności: {alpha}).")

# test normalności Kolmogorova-Smirnova
stat_ks, p_ks = stats.kstest(grupa_badana, 'norm', args=(grupa_badana.mean(), grupa_badana.std()))
test_norm = 'Kolmogorova-Smirnova'
print(f"\nStatystyka testu Kolmogorova-Smirnova: {stat_ks}")
print(f"p-wartość: {p_ks}")
if (p_ks < alpha):
    print(f"Test normalności {test_norm}: Odrzucamy hipotezę o normalności rozkładu (poziom istotności: {alpha}).")
else:
    print(f"Test normalności {test_norm}: Brak podstaw do odrzucenia hipotezy o normalności (poziom istotności: {alpha}).")

# ----- ANALIZA WARIANCJI -----
# test ANOVA (Welch’a)
stat_anova, p_anova = stats.f_oneway(grupa_1, grupa_2, grupa_3)
print(f"\nStatystyka testu ANOVA (Welch'a): {stat_anova}")
print(f"p-wartość: {p_anova}")
if (p_anova < alpha):
    print("Istnieją istotne różnice między grupami.")
else:
    print("Brak istotnych różnic między grupami.")

# test Kruskala-Wallisa
stat_kruskal, p_kruskal = stats.kruskal(grupa_1, grupa_2, grupa_3)
print(f"\nStatystyka testu Kruskala-Wallisa: {stat_kruskal}")
print(f"p-wartość: {p_kruskal}")
if (p_anova < p_kruskal):
    print("Istnieją istotne różnice między grupami.")
else:
    print("Brak istotnych różnic między grupami.")

# # Tworzenie histogramu
# plt.figure(figsize=(8, 5))
# # plt.hist(grupa_1, bins=10, alpha=0.5, label='Grupa 1', color='blue')
# # plt.hist(grupa_2, bins=10, alpha=0.5, label='Grupa 2', color='red')
# # plt.hist(grupa_3, bins=10, alpha=0.5, label='Grupa 3', color='green')
# plt.hist(dane['szczescie'], range(int(dane['szczescie'].min()), int(dane['szczescie'].max()) + 2), color='skyblue', edgecolor='black', alpha=0.7, align='left')
#
# # Ustawienie ticków osi X dla wszystkich wartości
# plt.xticks(range(int(dane['szczescie'].min()), int(dane['szczescie'].max()) + 1))
#
# # Tworzenie wykresu pudełkowego dla każdej grupy
# plt.figure(figsize=(8, 5))
# sns.boxplot(x=dane['grupa'], y=dane['szczescie'], palette='pastel')
#
# plt.xlabel('Szczęście')
# plt.ylabel('Częstotliwość')
# plt.title('Histogram poziomu szczęścia dla grup')
# plt.legend()
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()
