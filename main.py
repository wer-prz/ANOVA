from statsmodels.stats.power import FTestAnovaPower
from math import ceil
import pandas as pd
from scipy import  stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import boxcox, chisquare, skew, kurtosis, shapiro
pd.set_option('display.max_columns', None)  # Pokazuje wszystkie kolumny
pd.set_option('display.width', 1000)  # Zwiększa szerokość wyświetlania

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
# print("\nZliczanie wartości w kolumnie 'płeć':")
# print(dane['płeć'].value_counts())
# print("\nZliczanie wartości w kolumnie 'akt_fiz':")
# print(dane['akt_fiz'].value_counts())
# print("\nZliczanie wartości w kolumnie 'spot_tow':")
# print(dane['spot_tow'].value_counts())
# print("Zliczanie wartości w kolumnie 'ekran':")
# print(dane['ekran'].value_counts())
# print("\nZliczanie wartości w kolumnie 'sen':")
# print(dane['sen'].value_counts())
# print("\nZliczanie wartości w kolumnie 'choroby':")
# print(dane['choroby'].value_counts())
# print("\nZliczanie wartości w kolumnie 'szczescie':")
# print(dane['szczescie'].value_counts())


# Transformacje
dane['spot_tow_log'] = np.log(dane['spot_tow'] + 1)
dane['spot_tow_sqrt'] = np.sqrt(dane['spot_tow'])
dane['spot_tow_boxcox'], best_lambda = boxcox(dane['spot_tow'] + 1)
qt = QuantileTransformer(output_distribution='normal')
dane['spot_tow_qrt'] = qt.fit_transform(dane[['spot_tow']])

dane['ekran_log'] = np.log(dane['ekran'] + 1)
dane['ekran_sqrt'] = np.sqrt(dane['ekran'])
dane['ekran_boxcox'], best_lambda = boxcox(dane['ekran'] + 1)
qt = QuantileTransformer(output_distribution='normal')
dane['ekran_qrt'] = qt.fit_transform(dane[['ekran']])

dane['sen_log'] = np.log(dane['sen'] + 1)
dane['sen_sqrt'] = np.sqrt(dane['sen'])
dane['sen_boxcox'], best_lambda = boxcox(dane['sen'] + 1)
qt = QuantileTransformer(output_distribution='normal')
dane['sen_qrt'] = qt.fit_transform(dane[['sen']])

dane['choroby_log'] = np.log(dane['choroby'] + 1)
dane['choroby_sqrt'] = np.sqrt(dane['choroby'])
dane['choroby_boxcox'], best_lambda = boxcox(dane['choroby'] + 1)
qt = QuantileTransformer(output_distribution='normal')
dane['choroby_qrt'] = qt.fit_transform(dane[['choroby']])

dane['szczescie_log'] = np.log(dane['szczescie'] + 1)
dane['szczescie_sqrt'] = np.sqrt(dane['szczescie'])
dane['szczescie_boxcox'], best_lambda = boxcox(dane['szczescie'] + 1)
qt = QuantileTransformer(output_distribution='normal')
dane['szczescie_qrt'] = qt.fit_transform(dane[['szczescie']])

# rysowanie histogramów (dla zmienej ekran)
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
transformations = ['ekran', 'ekran_log', 'ekran_sqrt', 'ekran_boxcox', 'ekran_qrt']
titles = ['Oryginalne', 'Logarytmiczna', 'Pierwiastkowa', f'Box-Cox (λ={round(best_lambda,2)})', 'Kwantylowa']
for ax, trans, title in zip(axes, transformations, titles):
    ax.hist(dane[trans], bins=20, color='c', alpha=0.7, edgecolor='black')
    ax.set_title(title)
plt.tight_layout()
plt.show()

zmienna = 'spot_tow'

# Podział na grupy
grupa_1 = dane[dane['grupa'] == 1][zmienna]
grupa_2 = dane[dane['grupa'] == 2][zmienna]
grupa_3 = dane[dane['grupa'] == 3][zmienna]
grupa_badana = grupa_2

# Zlicz liczebności grup
liczebnosci = dane['grupa'].value_counts().sort_index()

# Test chi-kwadrat dla liczebności grup
stat, p_value = chisquare(liczebnosci)

# Wyniki testu
print(f"Statystyka testu chi-kwadrat: {stat}")
print(f"p-wartość: {p_value}")

# Interpretacja
alpha = 0.05  # Poziom istotności
if p_value < alpha:
    print("Grupy NIE są równoliczne na poziomie istotności 0.05.")
else:
    print("Grupy można uznać za równoliczne.")

print(zmienna)



# --------------------------------------------- TESTY NORMALNOŚCI ------------------------------------------------------
# test normalności Andersona-Darlinga
stat_anderson, krytyczne_wartosci, poziomy_istotnosci = stats.anderson(grupa_badana, dist='norm')
test_norm = 'Andersona-Darlinga'
print(f"\nStatystyka testu Andersona-Darlinga: {stat_anderson}")
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
stat_ks, p_ks = stats.kstest(grupa_1, 'norm', args=(grupa_1.mean(), grupa_1.std()))
test_norm = 'Kolmogorova-Smirnova'
print(f"\nStatystyka testu Kolmogorova-Smirnova: {stat_ks}")
print(f"p-wartość: {p_ks}")
if (p_ks < alpha):
    print(f"Test normalności {test_norm}: Odrzucamy hipotezę o normalności rozkładu (poziom istotności: {alpha}).")
else:
    print(f"Test normalności {test_norm}: Brak podstaw do odrzucenia hipotezy o normalności (poziom istotności: {alpha}).")



# ----------------------------------------- TESTY HOMOGENICZNOŚCI WARIANCJI --------------------------------------------
# test Levene'a - (bardziej odporny na niezgodność z normalnością):
stat_levene, p_levene = stats.levene(grupa_1, grupa_2, grupa_3)
test_wariancji = 'Levene’a'
print(f"\nStatystyka testu {test_wariancji}: {stat_levene}")
print(f"p-wartość: {p_levene}")
if p_levene < alpha:
    print(f"Test homogeniczności wariancji {test_wariancji}: Odrzucamy hipotezę o równości wariancji (poziom istotności: {alpha}).")
else:
    print(f"Test homogeniczności wariancji {test_wariancji}: Brak podstaw do odrzucenia hipotezy o równości wariancji (poziom istotności: {alpha}).")

# test Bartletta (lepszy dla normalnych danych, ale wrażliwy na odchylenia):
stat_bartlett, p_bartlett = stats.bartlett(grupa_1, grupa_2, grupa_3)
test_wariancji = 'Bartletta'
print(f"\nStatystyka testu {test_wariancji}: {stat_bartlett}")
print(f"p-wartość: {p_bartlett}")
if p_bartlett < alpha:
    print(f"Test homogeniczności wariancji {test_wariancji}: Odrzucamy hipotezę o równości wariancji (poziom istotności: {alpha}).")
else:
    print(f"Test homogeniczności wariancji {test_wariancji}: Brak podstaw do odrzucenia hipotezy o równości wariancji (poziom istotności: {alpha}).")

# test - Flignera-Killeena (odporny na brak normalności):
stat_fligner, p_fligner = stats.fligner(grupa_1, grupa_2, grupa_3)
test_wariancji = 'Flignera-Killeena'
print(f"\nStatystyka testu {test_wariancji}: {stat_fligner}")
print(f"p-wartość: {p_fligner}")
if p_fligner < alpha:
    print(f"Test homogeniczności wariancji {test_wariancji}: Odrzucamy hipotezę o równości wariancji (poziom istotności: {alpha}).")
else:
    print(f"Test homogeniczności wariancji {test_wariancji}: Brak podstaw do odrzucenia hipotezy o równości wariancji (poziom istotności: {alpha}).")



# --------------------------------------------- ANALIZA WARIANCJI ------------------------------------------------------
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

# # tworzenie histogramu
# plt.figure(figsize=(8, 5))
# plt.hist(grupa_1, bins=10, alpha=0.5, label='Grupa 1', color='blue')
# plt.hist(grupa_2, bins=10, alpha=0.5, label='Grupa 2', color='red')
# plt.hist(grupa_3, bins=8, alpha=0.5, label='Grupa 3', color='green')
# plt.hist(dane['szczescie'], range(int(dane['szczescie'].min()), int(dane['szczescie'].max()) + 2), color='skyblue', edgecolor='black', alpha=0.7, align='left')

Q1 = np.percentile(dane[zmienna], 25)  # Pierwszy kwartyl
Q3 = np.percentile(dane[zmienna], 75)  # Trzeci kwartyl
IQR = Q3 - Q1
dolna_granica = Q1 - 1.5 * IQR
górna_granica = Q3 + 1.5 * IQR
# wartości odstające:
outliers = dane[(dane[zmienna] < dolna_granica) | (dane[zmienna] > górna_granica)]
print(f"Wartości odstające: {outliers}")

print("Skośność:", skew(grupa_1), skew(grupa_2), skew(grupa_3))
print("Kurtoza:", kurtosis(grupa_1), kurtosis(grupa_2), kurtosis(grupa_3))

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
grupy = [grupa_1, grupa_2, grupa_3]
titles = ['grupa_1', 'grupa_2', 'grupa_3']

for ax, grupa, title in zip(axes, grupy, titles):
    s = skew(grupa)
    k = kurtosis(grupa)

    # histogram
    num_bins = int(np.ceil(np.log2(len(grupa)) + 1))
    ax.hist(grupa, bins=num_bins, density=True, color='c', alpha=0.7, edgecolor='black', label="Dane", align='mid')
    mean = np.mean(grupa)
    std_dev = np.std(grupa)

    # nałożenie teoretycznego rozkładu normalnego
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mean, std_dev)
    ax.plot(x, p, 'r', linewidth=2, label=f'Rozkład normalny\n$\mu={mean:.2f}$, $\sigma={std_dev:.2f}$')
    ax.set_title(f'{title}\nSkosność: {s:.2f}, Kurtoza: {k:.2f}')
    ax.legend()
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
grupy = [grupa_1, grupa_2, grupa_3]
titles = ['Grupa 1', 'Grupa 2', 'Grupa 3']

for ax, grupa, title in zip(axes, grupy, titles):
    stats.probplot(grupa, dist="norm", plot=ax)
    ax.set_title(f'Q-Q plot: {title}')

plt.tight_layout()
plt.show()

print(shapiro(grupa_1))
print(shapiro(grupa_2))
print(shapiro(grupa_3))

# wykresy pudełkowe dla grup
plt.figure(figsize=(8, 5))
sns.boxplot(x=dane['grupa'], y=dane[zmienna], palette='pastel')

plt.xlabel(zmienna)
plt.ylabel('Częstotliwość')
plt.title('Histogram poziomu szczęścia dla grup')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
