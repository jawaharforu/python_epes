from scipy import stats
data = stats.gamma.rvs(2, loc=4.733333E-01, scale=2.405318E-01, size=30)

from fitter import Fitter
f = Fitter(data, distributions=['norm', 't', 'triang', 'lognorm', 'uniform', 'expon', 'weibull_min',
                                'weibull_max','beta','gamma','logistic','pareto'])
f.fit()
#f.get_best()
f.summary()
from numpy import *
from matplotlib.pyplot import *
import scipy.stats

dist = scipy.stats.gamma
param = (f.fitted_param['gamma'])
print(param)
X = linspace(1.200000E-01,9.900000E-01, 30)
pdf_fitted = dist.pdf(X, *param)
print(X)
print(pdf_fitted)
plot(X, pdf_fitted, 'o')
show()

