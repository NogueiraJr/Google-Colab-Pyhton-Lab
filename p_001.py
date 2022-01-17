import numpy as np
import statsmodels.api as sm

rand_data = sm.datasets.randhie.load(as_pandas=False)
rand_exog = rand_data.exog.view(float).reshape(len(rand_data.exog), -1)
rand_exog = sm.add_constant(rand_exog, prepend=False)

poisson_mod = sm.Poisson(rand_data.endog, rand_exog)
poisson_res = poisson_mod.fit(method="newton")
print(poisson_res.summary())
