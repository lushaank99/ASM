import math
from pandas.tools.plotting import autocorrelation_plot

f = lambda x: dg.pdf(x) * math.pi

def mh_sampler(T=100):
    T=100
    x_curr = np.random.rand()
    while True:
        for i in range(T):
            x_next = np.random.normal(x_curr, 2)
            if min(1, f(x_next) / f(x_curr)) > np.random.uniform(0, 1):
                x_curr = x_next
        yield x_curr

sampler = mh_sampler()
for x in range(1000):
    next(sampler)

samples = [next(sampler) for x in range(10000)]

df['Target'].plot(color='blue', style='--', figsize=(8,6), linewidth=2.0)
pd.Series(samples).hist(bins=300, normed=True, color='green',
                        alpha=0.3, linewidth=0.0)
plt.legend(['Target PDF', 'MH Sampling'])
plt.show()

autocorrelation_plot(pd.Series(samples))