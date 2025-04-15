from statsmodels.stats.power import FTestAnovaPower
from math import ceil
import pandas as pd
from scipy import  stats
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import boxcox, chisquare, skew, kurtosis, shapiro
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# ------------------------------------- WYZNACZENIE WYMAGANEJ LICZEBNOŚCI GRUP -----------------------------------------
# # parametry
alpha = 0.05  # poziom istotności (prawdopodobieństwo popełnienia błędu 1 rodzaju)
# power = 0.8   # moc testu = 1 - beta, (beta - prawdopodobieństwo popełnienia błędu 2 rodzaju)
# groups = 3    # liczba grup
# # wskaźnik siły efektu f Cohena
# fCohen = 0.25   # niski = 0.10, średni = 0.25, wysoki = 0.40
#
# def group_size(alpha, power, groups, fCohen):
#     return ceil(FTestAnovaPower().solve_power(effect_size=fCohen, k_groups=groups, alpha=alpha, power=power))
#
# print("Alfa:", alpha, "\nMoc testu:", power, "\nLiczba grup:", groups, "\nWskaźnik f Cohena:", fCohen,
#       "\nLiczebność każdej grupy:", group_size(alpha, power, groups, fCohen),
#       "\nŁączna liczba badanych:", groups * group_size(alpha, power, groups, fCohen))

# ---------------------------------------- KONWERSJA DANYCH NA LICZBOWE ------------------------------------------------
#
dane = pd.read_excel('dane.xlsx')       # wczytanie pliku
dane = dane.iloc[:, 1:]                 # usunięcie pierwszej kolumny
dane.columns = ['wiek', 'płeć', 'akt_fiz', 'spot_tow', 'ekran', 'sen', 'choroby', 'szczęście']  # zmiana nazw kolumn
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

# print(dane[['wiek', 'płeć', 'akt_fiz', 'spot_tow', 'ekran', 'sen', 'choroby', 'szczęście']].describe(include='all'))
                                                                                             # opis statystyczny kolumn

# #--------------------------------------------- PODZIAŁ DANYCH NA GRUPY -------------------------------------------------

# definiowanie przedziałów
bins = [-1, 1, 4, 11]   # przedziały: 0-1h, 2-4h, >5h
labels = [1, 2, 3]      # oznaczenia grup

# przypisanie grup do nowej kolumny na podstawie przedziałów
dane['grupa'] = pd.cut(dane['akt_fiz'], bins=bins, labels=labels, right=True)

# zliczanie wartości w kolumnie 'grupa'
print(dane['grupa'].value_counts())

# # --------------------------------------------- TEST RÓWNOLICZNOŚCI GRUP -----------------------------------------------
#
# liczebnosci = dane['grupa'].value_counts().sort_index()
#
# # test chi-kwadrat dla liczebności grup
# stat, p_value = chisquare(liczebnosci)
#
# print(f"Statystyka testu chi-kwadrat: {stat}")
# print(f"p-wartość: {p_value}")
#
# # interpretacja
# if p_value < alpha:
#     print("Grupy NIE są równoliczne na poziomie istotności 0.05.")
# else:
#     print("Grupy można uznać za równoliczne.")
#
#-------------------------------------------------- TRANSFORMACJE ------------------------------------------------------
#
# # transformacje
# dane['szczęście_log'] = np.log(dane['szczęście'] + 1)
# dane['szczęście_sqrt'] = np.sqrt(dane['szczęście'])
dane['szczęście_boxcox'], best_lambda = boxcox(dane['szczęście'] + 1)
# qt = QuantileTransformer(output_distribution='normal')
# dane['szczęście_qrt'] = qt.fit_transform(dane[['szczęście']])

# ------------------------------------------------ WYBÓR ZMIENNEJ ------------------------------------------------------

zmienna = 'szczęście'

# podział na grupy
grupa_1 = dane[dane['grupa'] == 1][zmienna]
grupa_2 = dane[dane['grupa'] == 2][zmienna]
grupa_3 = dane[dane['grupa'] == 3][zmienna]
grupa_badana = grupa_3

# --------------------------------------------- TESTY NORMALNOŚCI ------------------------------------------------------
#
# # test normalności Kolmogorova-Smirnova
# stat_ks, p_ks = stats.kstest(grupa_badana, 'norm', args=(grupa_badana.mean(), grupa_badana.std()))
# test_norm = 'Kolmogorova-Smirnova'
# print(f"\nStatystyka testu Kolmogorova-Smirnova: {stat_ks}")
# print(f"p-wartość: {p_ks}")
# if (p_ks < alpha):
#     print(f"Odrzucamy hipotezę o normalności rozkładu (poziom istotności: {alpha}).")
# else:
#     print(f"Brak podstaw do odrzucenia hipotezy o normalności (poziom istotności: {alpha}).")
#
# # test normalności Andersona-Darlinga
# stat_anderson, krytyczne_wartosci, poziomy_istotnosci = stats.anderson(grupa_badana, dist='norm')
# test_norm = 'Andersona-Darlinga'
# print(f"\nStatystyka testu Andersona-Darlinga: {stat_anderson}")
# print(f"Krytyczne wartości: {krytyczne_wartosci}")
# print(f"Poziomy istotności: {poziomy_istotnosci}")
# wartosc_krytyczna = krytyczne_wartosci[2]
# if (stat_anderson > wartosc_krytyczna):
#     print(f"Odrzucamy hipotezę o normalności rozkładu (poziom istotności: {alpha}).")
# else:
#     print(f"Brak podstaw do odrzucenia hipotezy o normalności (poziom istotności: {alpha}).")
#
# # test normalności Shapiro-Wilka
# stat_shapiro, p_shapiro = stats.shapiro(grupa_badana)
# test_norm = 'Shapiro-Wilka'
# print(f"\nStatystyka testu Shapiro-Wilka: {stat_shapiro}")
# print(f"p-wartość: {p_shapiro}")
# if (p_shapiro < alpha):
#     print(f"Odrzucamy hipotezę o normalności rozkładu (poziom istotności: {alpha}).")
# else:
#     print(f"Brak podstaw do odrzucenia hipotezy o normalności (poziom istotności: {alpha}).")
#
# # test normalności Lillieforsa
# statystyka, p_l = lilliefors(grupa_badana)
# test_norm = 'Lillieforsa'
# print(f"\nStatystyka testu Lillieforsa:", statystyka)
# print(f"p-wartość:", p_l)
# if (p_l < alpha):
#     print(f"Odrzucamy hipotezę o normalności rozkładu (poziom istotności: {alpha}).")
# else:
#     print(f"Brak podstaw do odrzucenia hipotezy o normalności (poziom istotności: {alpha}).")

# ----------------------------------------- TESTY HOMOGENICZNOŚCI WARIANCJI --------------------------------------------
#
# # test Levene'a (bardziej odporny na niezgodność z normalnością):
# stat_levene, p_levene = stats.levene(grupa_1, grupa_2, grupa_3)
# test_wariancji = 'Levene’a'
# print(f"\nStatystyka testu {test_wariancji}: {stat_levene}")
# print(f"p-wartość: {p_levene}")
# if p_levene < alpha:
#     print("Odrzucamy hipotezę o równości wariancji.")
# else:
#     print("Brak podstaw do odrzucenia hipotezy o równości wariancji.")
#
# # test Bartletta (lepszy dla normalnych danych, ale wrażliwy na odchylenia):
# stat_bartlett, p_bartlett = stats.bartlett(grupa_1, grupa_2, grupa_3)
# test_wariancji = 'Bartletta'
# print(f"\nStatystyka testu {test_wariancji}: {stat_bartlett}")
# print(f"p-wartość: {p_bartlett}")
# if p_bartlett < alpha:
#     print("Odrzucamy hipotezę o równości wariancji.")
# else:
#     print("Brak podstaw do odrzucenia hipotezy o równości wariancji.")
#
# # test Flignera-Killeena (odporny na brak normalności):
# stat_fligner, p_fligner = stats.fligner(grupa_1, grupa_2, grupa_3)
# test_wariancji = 'Flignera-Killeena'
# print(f"\nStatystyka testu {test_wariancji}: {stat_fligner}")
# print(f"p-wartość: {p_fligner}")
# if p_fligner < alpha:
#     print("Odrzucamy hipotezę o równości wariancji.")
# else:
#     print("Brak podstaw do odrzucenia hipotezy o równości wariancji.")
#
#
# # --------------------------------------------- ANALIZA WARIANCJI ----------------------------------------------------

# test ANOVA (Fisher’a)
stat_anova, p_anova = stats.f_oneway(grupa_1, grupa_2, grupa_3)
print(f"\nStatystyka testu ANOVA (Fisher'a): {stat_anova}")
print(f"p-wartość: {p_anova}")
if (p_anova < alpha):
    print("Istnieją istotne różnice między grupami.")
else:
    print("Brak istotnych różnic między grupami.")

# test Welch'a
wyniki = pg.welch_anova(dv=zmienna, between='grupa', data=dane)
f_stat = wyniki['F'].values[0]
p_val = wyniki['p-unc'].values[0]
df_between = wyniki['ddof1'].values[0]
df_within = wyniki['ddof2'].values[0]
print(f"\nStatystyka testu Welch'a ANOVA: F({df_between:.0f}, {df_within:.2f}) = {f_stat:.3f}")
print(f"p-wartość: {p_val:.4f}")
if (p_val < alpha):
    print("Istnieją istotne różnice między grupami.")
else:
    print("Brak istotnych różnic między grupami.")

# test Kruskala-Wallisa
stat_kruskal, p_kruskal = stats.kruskal(grupa_1, grupa_2, grupa_3)
print(f"\nStatystyka testu Kruskala-Wallisa: {stat_kruskal}")
print(f"p-wartość: {p_kruskal}")
if (p_kruskal < alpha):
    print("Istnieją istotne różnice między grupami.")
else:
    print("Brak istotnych różnic między grupami.")

# ------------------------------------------------ HISTOGRAMY ----------------------------------------------------------
# tworzenie histogramu
# # plt.figure(figsize=(8, 5))
# # plt.hist(grupa_1, bins=10, alpha=0.5, label='Grupa 1', color='blue')
# # plt.hist(grupa_2, bins=10, alpha=0.5, label='Grupa 2', color='red')
# # plt.hist(grupa_3, bins=8, alpha=0.5, label='Grupa 3', color='green')
# # plt.hist(dane[zmienna], range(int(dane[zmienna].min()), int(dane[zmienna].max()) + 2), color='skyblue', edgecolor='black', alpha=0.7, align='left')
#
# fig, axes = plt.subplots(2,2, figsize=(11, 6))
# grupy = [dane[zmienna], grupa_1, grupa_2, grupa_3]
# titles = [zmienna, 'grupa_1', 'grupa_2', 'grupa_3']
#
# axes = axes.flatten()
#
# xmin = dane[zmienna].min()
# xmax = dane[zmienna].max()
# print(xmax, xmin)
# bins = np.linspace(xmin, xmax, 17)
# x_range = np.linspace(xmin, xmax, 100)
#
# for ax, grupa, title in zip(axes, grupy, titles):
#     s = skew(grupa)
#     k = kurtosis(grupa)
#
#     ax.hist(grupa, bins=bins, density=True, color='c', alpha=1, edgecolor='black', label="Dane", align='mid')
#     mean = np.mean(grupa)
#     std_dev = np.std(grupa)
#
#     # nałożenie teoretycznego rozkładu normalnego
#     xmin, xmax = ax.get_xlim()
#     x = np.linspace(xmin, xmax, 100)
#     p = stats.norm.pdf(x, mean, std_dev)
#     ax.plot(x, p, 'r', linewidth=2, label=f'Rozkład normalny\n$\mu={mean:.2f}$, $\sigma={std_dev:.2f}$')
#     ax.set_xlim(xmin, xmax)
#     ax.set_xticks(bins)
#     ax.set_title(f'{title}\nSkosność: {s:.2f}, Kurtoza: {k:.2f}')
#     ax.legend()
#
# plt.tight_layout()
# plt.savefig("szczęście_qrt.pdf", format='pdf', bbox_inches='tight')
# plt.show()
#
# # ------------------------------------------- WYKRESY Q-Q --------------------------------------------------------------
#
# fig, axes = plt.subplots(2,2, figsize=(10, 6))
# grupy = [dane[zmienna], grupa_1, grupa_2, grupa_3]
# titles = [zmienna, 'Grupa 1', 'Grupa 2', 'Grupa 3']
#
# axes = axes.flatten()
#
# for ax, grupa, title in zip(axes, grupy, titles):
#     stats.probplot(grupa, dist="norm", plot=ax)
#     ax.set_xlabel("Kwantyle teoretyczne")
#     ax.set_ylabel("Kwantyle empiryczne")
#     ax.set_title(f'Q-Q plot: {title}')
#
# plt.tight_layout()
# plt.savefig("qq-szczęście_qrt.pdf", format='pdf', bbox_inches='tight')
# plt.show()
#
# # -------------------------------------- WYKRESY PUDEŁKOWE ----------------------------------------------------------
#
# # wykresy pudełkowe dla grup
# plt.figure(figsize=(8, 5))
# sns.boxplot(x=dane['grupa'], y=dane[zmienna], palette='pastel')
#
# plt.xlabel(zmienna)
# plt.ylabel('Częstotliwość')
# plt.title('Histogram poziomu szczęścia dla grup')
# plt.legend()
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()
