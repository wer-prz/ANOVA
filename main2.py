from statsmodels.stats.power import FTestAnovaPower
import statsmodels.stats.power as smp
from tabulate import tabulate
from math import ceil
import pandas as pd
from scipy import  stats
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import statsmodels.api as sm
import matplotlib.ticker as ticker
from sklearn.preprocessing import QuantileTransformer
from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import boxcox, chisquare, skew, kurtosis, shapiro
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# ------------------------------------- WYZNACZENIE WYMAGANEJ LICZEBNOŚCI GRUP -----------------------------------------
# parametry
alpha = 0.01   # poziom istotności
power = 0.8    # moc testu
groups = 3     # liczba grup
fCohen = 0.25  # wskaźnik siły efektu f Cohena
               # niski = 0.10, średni = 0.25, wysoki = 0.40

def group_size(alpha, power, groups, fCohen):
    return ceil(FTestAnovaPower().solve_power(effect_size=fCohen,
                        k_groups=groups, alpha=alpha, power=power))

print("Alfa:", alpha, "\nMoc testu:", power,
      "\nLiczba grup:", groups, "\nWskaźnik f Cohena:", fCohen,
      "\nLiczebność każdej grupy:",
            group_size(alpha, power, groups, fCohen),
      "\nŁączna liczba badanych:",
            groups * group_size(alpha, power, groups, fCohen))

# ---------------------------------------- KONWERSJA DANYCH NA LICZBOWE ------------------------------------------------
#
dane = pd.read_excel('dane_de.xlsx')     # wczytanie pliku
dane = dane.iloc[:, 1:]                  # usunięcie pierwszej kolumny
dane.columns = ['miasto', 'rok', 'pm10'] # zmiana nazw kolumn
dane = dane[dane['rok'] >= 2017]         # usunięcie wierszy, w których rok < 2017
dane = dane[dane['rok'] <= 2019]         # usunięcie wierszy, w których rok > 2019
dane = dane.reset_index(drop=True)       # resetowanie indeksów


# #--------------------------------------------- PODZIAŁ DANYCH NA GRUPY -------------------------------------------------

# definiowanie przedziałów
bins = [2016, 2017, 2018, 2019]  # przedziały: 2017, 2018, 2019
labels = [2017, 2018, 2019]      # oznaczenia grup

# przypisanie grup do nowej kolumny na podstawie przedziałów
dane['grupa'] = pd.cut(dane['rok'], bins=bins, labels=labels, right=True)

print(dane)

# zliczanie wartości w kolumnie 'grupa'
print(dane['grupa'].value_counts())

print(dane[['miasto', 'rok', 'pm10', 'grupa']].describe(include='all'))  # opis statystyczny kolumn

# # # --------------------------------------------- TEST RÓWNOLICZNOŚCI GRUP -----------------------------------------------

liczebnosci = dane['grupa'].value_counts().sort_index()

# test chi-kwadrat dla liczebności grup
stat, p_value = chisquare(liczebnosci)

print(f"\nStatystyka testu chi-kwadrat: {stat}")
print(f"p-wartość: {p_value}")

# interpretacja
if p_value < alpha:
    print("Grupy NIE są równoliczne na poziomie istotności 0.05.")
else:
    print("Grupy można uznać za równoliczne.")

# ------------------------------------------------ WYBÓR ZMIENNEJ ------------------------------------------------------
zmienna = 'pm10'

# podział na grupy
grupa_1 = dane[dane['grupa'] == 2017][zmienna]
grupa_2 = dane[dane['grupa'] == 2018][zmienna]
grupa_3 = dane[dane['grupa'] == 2019][zmienna]

#
# # --------------------------------------------- TESTY NORMALNOŚCI ------------------------------------------------------

# test normalności Kolmogorova-Smirnova

grupa_badana = grupa_1
stat_ks, p_ks = stats.kstest(grupa_badana, 'norm',
                             args=(grupa_badana.mean(), grupa_badana.std()))
print(f"\n2017 - Statystyka testu Kolmogorova-Smirnova: {stat_ks}")
print(f"p-wartość: {p_ks}")
if (p_ks < alpha):
    print(f"Odrzucamy hipotezę o normalności rozkładu.")
else:
    print(f"Brak podstaw do odrzucenia hipotezy o normalności.")

grupa_badana = grupa_2
stat_ks, p_ks = stats.kstest(grupa_badana, 'norm',
                             args=(grupa_badana.mean(), grupa_badana.std()))
print(f"\n2018 - Statystyka testu Kolmogorova-Smirnova: {stat_ks}")
print(f"p-wartość: {p_ks}")
if (p_ks < alpha):
    print(f"Odrzucamy hipotezę o normalności rozkładu.")
else:
    print(f"Brak podstaw do odrzucenia hipotezy o normalności.")

grupa_badana = grupa_3
stat_ks, p_ks = stats.kstest(grupa_badana, 'norm',
                             args=(grupa_badana.mean(), grupa_badana.std()))
print(f"\n2019 - Statystyka testu Kolmogorova-Smirnova: {stat_ks}")
print(f"p-wartość: {p_ks}")
if (p_ks < alpha):
    print(f"Odrzucamy hipotezę o normalności rozkładu.")
else:
    print(f"Brak podstaw do odrzucenia hipotezy o normalności.")

# ----------------------------------------- TESTY HOMOGENICZNOŚCI WARIANCJI --------------------------------------------

# test Levene'a (bardziej odporny na niezgodność z normalnością):
stat_levene, p_levene = stats.levene( grupa_1, grupa_2, grupa_3)
test_wariancji = 'Levene’a'
print(f"\nStatystyka testu {test_wariancji}: {stat_levene}")
print(f"p-wartość: {p_levene}")
if p_levene < alpha:
    print("Odrzucamy hipotezę o równości wariancji.")
else:
    print("Brak podstaw do odrzucenia hipotezy o równości wariancji.")

# test Bartletta (lepszy dla normalnych danych, ale wrażliwy na odchylenia):
stat_bartlett, p_bartlett = stats.bartlett(grupa_1, grupa_2, grupa_3)
test_wariancji = 'Bartletta'
print(f"\nStatystyka testu {test_wariancji}: {stat_bartlett}")
print(f"p-wartość: {p_bartlett}")
if p_bartlett < alpha:
    print("Odrzucamy hipotezę o równości wariancji.")
else:
    print("Brak podstaw do odrzucenia hipotezy o równości wariancji.")

# test Flignera-Killeena (odporny na brak normalności):
stat_fligner, p_fligner = stats.fligner(grupa_1, grupa_2, grupa_3)
test_wariancji = 'Flignera-Killeena'
print(f"\nStatystyka testu {test_wariancji}: {stat_fligner}")
print(f"p-wartość: {p_fligner}")
if p_fligner < alpha:
    print("Odrzucamy hipotezę o równości wariancji.")
else:
    print("Brak podstaw do odrzucenia hipotezy o równości wariancji.")


# # --------------------------------------------- ANALIZA WARIANCJI ----------------------------------------------------

# test ANOVA (Fisher’a)
stat_anova, p_anova = stats.f_oneway(grupa_1, grupa_2, grupa_3)
print(f"\nStatystyka testu ANOVA (Fishera): {stat_anova}")
print(f"p-wartość: {p_anova}")
if (p_anova < alpha):
    print("Istnieją istotne różnice między grupami.")
else:
    print("Brak istotnych różnic między grupami.")


means = [np.mean(grupa_1), np.mean(grupa_2), np.mean(grupa_3)]
grand_mean = np.mean(np.concatenate([grupa_1, grupa_2, grupa_3]))

variance_1 = np.var(grupa_1, ddof=1)
variance_2 = np.var(grupa_2, ddof=1)
variance_3 = np.var(grupa_3, ddof=1)

df_between = groups - 1
df_within = len(grupa_1) + len(grupa_2) + len(grupa_3) - 3
df_total = len(grupa_1) + len(grupa_2) + len(grupa_3) - 1

ss_between = sum(len(grupa_1) * (mean - grand_mean)**2 for mean in means)
ss_within = ((variance_1 * (len(grupa_1) - 1))
            + (variance_2 * (len(grupa_2) - 1))
            + (variance_3 * (len(grupa_3) - 1)))
ss_total = ss_between + ss_within

ms_between = ss_between / df_between
ms_within = ss_within / df_within

f_stat = ms_between / ms_within
effect_size_f = np.sqrt(ss_between / ss_total)

stat_anova, p_anova = stats.f_oneway(grupa_1, grupa_2, grupa_3)

anova_power = smp.FTestAnovaPower()
total_n = len(grupa_1) + len(grupa_2) + len(grupa_3)
power_anova = anova_power.power(effect_size=effect_size_f,
                k_groups=groups, nobs=total_n, alpha=alpha)

print(f"\nMoc testu = {power_anova}")

table = [["Pomiędzy grupami", f"{ss_between:.2f}", df_between,
          f"{ms_between:.2f}", f"{f_stat:.2f}", f"{p_anova:.4f}",
          f"{effect_size_f:.3f}", f"{power_anova:.4f}"],
        ["Wewnątrz grup", f"{ss_within:.2f}", df_within,
         f"{ms_within:.2f}", "", ""],
        ["Całkowita", f"{ss_total:.2f}", df_total, "", "", ""]]

headers = ["Źródło zmienności", "Suma kwadratów (SS)",
           "Stopnie swobody (df)", "Średni kwadrat (MS)",
           "F", "p-wartość", "f Cohena", "Moc"]

print(tabulate(table, headers=headers, tablefmt="grid"))

# Tukey HSD
print("Tukey HSD:")
tukey = pairwise_tukeyhsd(endog=dane['pm10'], groups=dane['grupa'], alpha=alpha)
print(tukey)

# NIR (LSD)
print("\nTest NIR (LSD):")
posthoc_nir = sp.posthoc_ttest(dane, val_col='pm10', group_col='grupa', p_adjust=None)
print(posthoc_nir)

# # ------------------------------------------------ HISTOGRAMY ----------------------------------------------------------
# tworzenie histogramu
fig, axes = plt.subplots(2,2, figsize=(10, 6))
grupy = [dane[zmienna], grupa_1, grupa_2, grupa_3]
titles = [zmienna, '2017', '2018', '2019']

axes = axes.flatten()

xmin = dane[zmienna].min()
xmax = dane[zmienna].max()
print(xmax, xmin)
bins = np.linspace(xmin, xmax, 50)
x_range = np.linspace(xmin, xmax, 100)

for ax, grupa, title in zip(axes, grupy, titles):
    s = skew(grupa)
    k = kurtosis(grupa)

    ax.hist(grupa, bins=bins, density=True, color='c', alpha=1, edgecolor='black', label="Dane", align='mid')
    mean = np.mean(grupa)
    std_dev = np.std(grupa)

    # nałożenie teoretycznego rozkładu normalnego
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mean, std_dev)
    ax.plot(x, p, 'r', linewidth=2, label=f'Rozkład normalny\n$\mu={mean:.2f}$, $\sigma={std_dev:.2f}$')
    ax.set_xlim(xmin, xmax)
    tick_step = 2  # możesz zmienić na inny krok
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_step))
    ax.set_title(f'{title}\nSkosność: {s:.2f}, Kurtoza: {k:.2f}')
    ax.legend()

plt.tight_layout()
plt.savefig("pm10.pdf", format='pdf', bbox_inches='tight')
plt.show()

# ------------------------------------------- WYKRESY Q-Q --------------------------------------------------------------

fig, axes = plt.subplots(2,2, figsize=(10, 6))
grupy = [dane[zmienna], grupa_1, grupa_2, grupa_3]
titles = [zmienna, '2017', '2018', '2019']

axes = axes.flatten()

for ax, grupa, title in zip(axes, grupy, titles):
    stats.probplot(grupa, dist="norm", plot=ax)
    ax.set_xlabel("Kwantyle teoretyczne")
    ax.set_ylabel("Kwantyle empiryczne")
    ax.set_title(f'Q-Q plot: {title}')

plt.tight_layout()
plt.savefig("gg-pm10.pdf", format='pdf', bbox_inches='tight')
plt.show()

# # -------------------------------------- WYKRESY PUDEŁKOWE ----------------------------------------------------------
#
# wykresy pudełkowe dla grup
plt.figure(figsize=(8, 5))
sns.boxplot(x=dane['grupa'], y=dane[zmienna], palette='pastel')

plt.xlabel('grupa')
plt.ylabel('poziom PM_10')
plt.title('Wykres pudełkowy poziomu pyłów zawieszonych PM_10 w Niemczech latach 2017-2019')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("boxplot-pm10.pdf", format='pdf', bbox_inches='tight')
plt.show()
