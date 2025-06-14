import pandas as pd
from scipy.stats import shapiro
from scipy.stats import levene
from scipy.stats import chisquare
from scipy.stats import kstest, zscore, anderson
from scipy.stats import stats
from scipy.stats import f_oneway
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
from statsmodels.stats.power import FTestAnovaPower
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp


# Parametry analizy mocy
alpha = 0.05        # poziom istotności
power = 0.8         # pożądana moc testu
k_groups = 3        # liczba grup
effect_size = 0.25  # wskaźnik f Cohena (średni efekt)

# Oblicz wymaganą liczność w każdej grupie
analysis = FTestAnovaPower()
sample_size_per_group = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha, k_groups=k_groups)
sample_size_per_group = int(np.ceil(sample_size_per_group))
total_sample_size = sample_size_per_group * k_groups

# Wyświetlenie wyników
print("Wyznaczenie wymaganej liczebności grup w Pythonie z użyciem funkcji solve_power z biblioteki statsmodels.stats.power:")
print(f"\nAlfa: {alpha}")
print(f"Moc testu: {power}")
print(f"Liczba grup: {k_groups}")
print(f"Wskaźnik f Cohena: {effect_size}")
print(f"Liczebność każdej grupy: {sample_size_per_group}")
print(f"Łączna liczba badanych: {total_sample_size}")


# 1. Wczytanie pliku CSV
df = pd.read_csv('Powerlifters.csv')

# 2. Filtrowanie
df = df[
    (df['Sex'] == 'F') &
    (df['Equipment'] == 'Raw') &
    (df['Division'] == 'Pro Open')
].copy()

# 3. Tworzenie kolumny STBR (squat to bench ratio)
df['STBR'] = df['Best3SquatKg'] / df['Best3BenchKg']

# 4. Usuwanie rekordów, gdzie STBR lub Age są puste lub Best3BenchKg == 0
df = df[
    df['STBR'].notna() &
    df['Age'].notna() &
    df['Best3SquatKg'].notna() &
    df['Best3BenchKg'].notna() &
    (df['Best3BenchKg'] != 0)
].copy()

# 5. Grupowanie po imieniu zawodnika i obliczanie średniego STBR i wieku
df_clean = df.groupby('Name').agg({
    'STBR': 'mean',
    'Age': 'mean'
}).reset_index()

# 6. Grupowanie wiekowe
g1 = 27
g2 = 32

def przypisz_grupe_wiekowa(wiek):
    if wiek <= g1:
        return 'Młodsi'
    elif wiek <= g2:
        return 'Średni'
    else:
        return 'Starsi'

df_clean['GrupaWiekowa'] = df_clean['Age'].apply(przypisz_grupe_wiekowa)

# 7. Podział na grupy
df1 = df_clean[df_clean['GrupaWiekowa'] == 'Młodsi']
df2 = df_clean[df_clean['GrupaWiekowa'] == 'Średni']
df3 = df_clean[df_clean['GrupaWiekowa'] == 'Starsi']



#top10 = df_clean.sort_values(by='STBR', ascending=False).head(10)
print(df_clean.head(5))
print(df_clean.tail(5))
# TEST LICZNOŚCI
print(f"Młodsi: {len(df1)}")
print(f"Średni: {len(df2)}")
print(f"Starsi: {len(df3)}")

# Test chi-kwadrat: porównujemy do oczekiwanej równej liczności

licznosci = df_clean['GrupaWiekowa'].value_counts().sort_index()
stat, p = chisquare(f_obs=licznosci)

print(f"\nTest chi² dobroci dopasowania:")
print(f"  Statystyka: {round(stat, 4)}")
print(f"  p-value:    {round(p, 4)}")

if p >= 0.05:
    print(" p>=0.05 - Brak podstaw do odrzucenia hipotezy – liczności są zbliżone.")
else:
    print(" p<0.05 - Różnice w liczności grup są istotne – mogą naruszać założenia.")

# --- TEST NORMALNOŚCI ---

# Kolmogorov-Smirnov
print("\nTest Kolmogorova-Smirnova:")
for grupa, dane in zip(['Młodsi', 'Średni', 'Starsi'], [df1, df2, df3]):
    z = zscore(dane['STBR'])
    stat, p = kstest(z, 'norm')
    print(f"{grupa}: stat = {round(stat, 4)}, p = {round(p, 4)}")
    if p >= 0.05:
        print(" p>=0.05 - Brak podstaw do odrzucenia H₀ – rozkład może być normalny.\n")
    else:
        print(" p<0.05 - Odrzucamy H₀ – rozkład odbiega od normalnego.\n")

# Shapiro-Wilk
print("\nTest Shapiro-Wilka (H0: rozkład normalny):")

for grupa, dane in zip(['Młodsi', 'Średni', 'Starsi'], [df1, df2, df3]):
    stat, p = shapiro(dane['STBR'])
    print(f"{grupa}: statystyka = {round(stat, 4)}, p-value = {round(p, 4)}")

    if p >= 0.05:
        print(" p>=0.05 - Brak podstaw do odrzucenia H₀ – rozkład może być normalny.\n")
    else:
        print(" p<0.05 - Odrzucamy H₀ – rozkład odbiega od normalnego.\n")

# Anderson-Darling
print("\nTest Andersona-Darlinga:")
for grupa, dane in zip(['Młodsi', 'Średni', 'Starsi'], [df1, df2, df3]):
    wynik = anderson(dane['STBR'], dist='norm')
    stat = wynik.statistic
    granica = wynik.critical_values[2]  # poziom 5%
    print(f"{grupa}: stat = {round(stat, 4)}, granica (5%) = {round(granica, 4)}")
    if p >= 0.05:
        print(" p>=0.05 - Brak podstaw do odrzucenia H₀ – rozkład może być normalny.\n")
    else:
        print(" p<0.05 - Odrzucamy H₀ – rozkład odbiega od normalnego.\n")

# Test Levene’a – weryfikacja jednorodności wariancji
print("\nTest Levene’a (H0: wariancje są jednorodne):")

stat, p = levene(df1['STBR'], df2['STBR'], df3['STBR'])

print(f"Statystyka testu = {round(stat, 4)}, p-value = {round(p, 4)}")

if p >= 0.05:
    print(" p>=0.05 - Brak podstaw do odrzucenia H₀ – wariancje można uznać za jednorodne.")
else:
    print(" p<0.05 - Odrzucamy H₀ – istnieją istotne różnice w wariancjach między grupami.")

# --- WYKRESY ---

plt.figure(figsize=(15, 4))
for i, (nazwa, dane) in enumerate(zip(['Młodsi', 'Średni', 'Starsi'], [df1, df2, df3])):
    plt.subplot(1, 3, i + 1)
    sns.histplot(dane['STBR'], kde=True, bins=30, color='cornflowerblue')
    plt.title(nazwa)
    plt.xlabel('STBR (Squat / Bench)')
    plt.ylabel('Liczba zawodników')

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(data=df_clean, x='GrupaWiekowa', y='STBR', palette='Set2')
plt.title("Boxplot STBR w grupach wiekowych")
plt.xlabel("Grupa wiekowa")
plt.ylabel("STBR (Squat / Bench)")
plt.tight_layout()
plt.show()

# ANOVA
stat, p = f_oneway(df1['STBR'], df2['STBR'], df3['STBR'])

print("\n--- Analiza wariancji ANOVA (jednoczynnikowa) ---")
print(f"Statystyka F = {round(stat, 4)}")
print(f"p-value      = {round(p, 4)}")

if p >= 0.05:
    print(" p >= 0.05 - Brak podstaw do odrzucenia H₀ – średnie STBR są podobne między grupami.\n")
else:
    print(" p < 0.05 - Odrzucamy H₀ – istnieją istotne różnice średnich STBR między grupami.\n")


# --- Ręczne obliczenie tabeli ANOVA i efektu f Cohena ---

# Lista grup
grupy = ['Młodsi', 'Średni', 'Starsi']
dane_grup = [df1['STBR'], df2['STBR'], df3['STBR']]

# Liczności i średnie
n_grupy = [len(g) for g in dane_grup]
mean_grupy = [np.mean(g) for g in dane_grup]

# Średnia ogólna
all_data = pd.concat(dane_grup)
mean_total = np.mean(all_data)

# Suma kwadratów pomiędzy grupami (SS_between)
ss_between = sum(n * (mean - mean_total) ** 2 for n, mean in zip(n_grupy, mean_grupy))

# Suma kwadratów wewnątrz grup (SS_within)
ss_within = sum(sum((x - mean) ** 2 for x in g) for g, mean in zip(dane_grup, mean_grupy))

# Stopnie swobody
df_between = len(dane_grup) - 1
df_within = sum(n_grupy) - len(dane_grup)
df_total = df_between + df_within

# Średnie kwadraty
ms_between = ss_between / df_between
ms_within = ss_within / df_within

# Statystyka F i p-value
f_stat = ms_between / ms_within
p_value = 1 - f.cdf(f_stat, df_between, df_within)

# Efekt f Cohena
f_cohen = np.sqrt(f_stat * df_between / df_within)

# Tabelka
anova_df = pd.DataFrame({
    'Źródło zmienności': ['Pomiędzy grupami', 'Wewnątrz grup', 'Ogółem'],
    'Suma kwadratów (SS)': [ss_between, ss_within, ss_between + ss_within],
    'Stopnie swobody (df)': [df_between, df_within, df_total],
    'Średni kwadrat (MS)': [ms_between, ms_within, ''],
    'F': [f_stat, '', ''],
    'p-value': [p_value, '', ''],
    'f Cohena': [f_cohen, '', '']
})

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)
pd.set_option('display.expand_frame_repr', False)


print(anova_df)

# Parametry:
effect_size = f_cohen
alpha = 0.05
nobs = sum(n_grupy)
k_groups = len(n_grupy)

# Oblicz moc testu (power)
power = FTestAnovaPower().power(effect_size=effect_size, nobs=nobs, alpha=alpha, k_groups=k_groups)

print(f"\n Moc testu ANOVA: {round(power, 4)}")

df_display = df_clean[['Name', 'Age', 'STBR', 'GrupaWiekowa']]

# Wyświetlenie 5 pierwszych
print("🔹 Pierwsze 5 rekordów:\n")
print(df_display.head())

# Wyświetlenie 5 ostatnich
print("\n🔹 Ostatnie 5 rekordów:\n")
print(df_display.tail())

# Przygotowanie danych: wartości STBR i grupy wiekowe
stbr_values = df_clean['STBR']
groups = df_clean['GrupaWiekowa']

print("\n--- Test Post-Hoc: Tukey HSD ---")
tukey = pairwise_tukeyhsd(endog=stbr_values, groups=groups, alpha=0.05)
print(tukey)

print("\n--- Test Post-Hoc: NIR / LSD (Test najmniejszych istotnych różnic) ---")
nir = sp.posthoc_ttest(df_clean, val_col='STBR', group_col='GrupaWiekowa', p_adjust='holm')
print(nir)