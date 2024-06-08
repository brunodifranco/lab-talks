from pandas.core.series import Series
from statsmodels.tsa.stattools import adfuller

def adf_test(series: Series):
    result = adfuller(series)
    print(f'p-value ({series.name}): {result[1]:.6f}')

    if result[1] < 0.05:
        print(f'Rejeitamos a hipótese nula de que a série {series.name} tem uma raiz unitária. Portanto, a série {series.name} é estacionária.')
    else:
        print(f'Não rejeitamos a hipótese nula de que a série {series.name} tem uma raiz unitária. Portanto, a série {series.name} não é estacionária.')