from statsmodels.stats.power import FTestAnovaPower
from math import ceil

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
