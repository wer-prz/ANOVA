import pandas as pd
from scipy.stats import shapiro
from scipy.stats import levene
from scipy.stats import chisquare
from scipy.stats import kstest, zscore, anderson
from scipy.stats import stats
import seaborn as sns
import matplotlib.pyplot as plt


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



top10 = df_clean.sort_values(by='STBR', ascending=False).head(10)
print(top10)

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