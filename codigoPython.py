import pingouin as pg
import pandas as pd
import numpy as np

# 1) Datos (ejemplo con 12 sujetos por grupo)
np.random.seed(42)
grupo_A = np.random.normal(loc=88, scale=4, size=12)  # media 88, sd ≈4
grupo_B = np.random.normal(loc=82, scale=5, size=12)  # media 82, sd ≈5

data = pd.DataFrame({
    'Grupo': ['A']*12 + ['B']*12,
    'Nota' : np.concatenate([grupo_A, grupo_B])
})

# 2) Estadística descriptiva por grupo
desc = data.groupby('Grupo')['Nota'].describe()
print("Descriptivos:\n", desc)

# 3) Normalidad por grupo (Shapiro-Wilk)
norm = pg.normality(data, dv='Nota', group='Grupo')
print("\nNormalidad (Shapiro-W):\n", norm)

# 4) Homocedasticidad (Levene)
hom = pg.homoscedasticity(data, dv='Nota', group='Grupo', method='levene')
print("\nHomoscedasticidad (Levene):\n", hom)

# 5) Elegir y aplicar prueba:
# Si ambos grupos normales y varianzas iguales -> ttest (paired=False)
# Si no normales -> mwu (Mann-Whitney U)
if norm['normal'].all() and hom['equal_var'].all():
    res = pg.ttest(data[data['Grupo']=='A']['Nota'],
                data[data['Grupo']=='B']['Nota'],
                paired=False, correction='auto')
    test_name = 'T-test (independiente)'
else:
    res = pg.mwu(data[data['Grupo']=='A']['Nota'],
                data[data['Grupo']=='B']['Nota'])
    test_name = 'Mann-Whitney U'

print(f"\n{test_name}:\n", res)

# 6) Tamaño del efecto (Cohen's d)
d = pg.compute_effsize(data[data['Grupo']=='A']['Nota'],
                    data[data['Grupo']=='B']['Nota'],
                    eftype='cohen', paired=False)
print("\nCohen's d:", round(d, 3))
