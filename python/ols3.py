import numpy as np 
import pandas as pd 
import statsmodels.api as sm 
import statsmodels.formula.api as smf 
from itertools import combinations 
import plotnine as p

# read data
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
def read_data(file): 
	return pd.read_stata("https://github.com/scunning1975/mixtape/raw/master/" + file)

coefs = np.zeros(1000)
for i in range(1000):
    tb = pd.DataFrame({
    'x': 9*np.random.normal(size=10000),
    'u': 36*np.random.normal(size=10000)})
    tb['y'] = 3 + 2 * tb['x'].to_numpy() + tb['u'].to_numpy()

    reg_tb = sm.OLS.from_formula('y ~ x', data=tb).fit()

    coefs[i] = reg_tb.params['x']

print(reg_tb.params)
print(reg_tb.tvalues)
print(reg_tb.t_test([1, 0]))
print(reg_tb.f_test(np.identity(2)))
print(f"coefs:\n{coefs.shape}")
(
    p.ggplot() +
    p.geom_histogram(p.aes(x=coefs), binwidth = 0.01)
).show()
