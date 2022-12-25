def crude_monte_carlo(num_samples=1000):
    lower_bound = 0
    upper_bound = 5

    sum_of_samples = 0
    for i in range(num_samples):
        x = get_rand_number(lower_bound, upper_bound)
        sum_of_samples += f_of_x(x) #This is returns value of function f(x) we observe

    return (upper_bound - lower_bound) * float(sum_of_samples / num_samples)


def get_crude_MC_variance(num_samples):
    int_max = 5  # this is the max of our integration range

    # get the average of squares
    running_total = 0
    for i in range(num_samples):
        x = get_rand_number(0, int_max)
        running_total += f_of_x(x) ** 2
    sum_of_sqs = running_total * int_max / num_samples

    # get square of average
    running_total = 0
    for i in range(num_samples):
        x = get_rand_number(0, int_max)
        running_total = f_of_x(x)
    sq_ave = (int_max * running_total / num_samples) ** 2

    return sum_of_sqs - sq_ave

def inverse_G_of_r(r, lamda):
    return (-1 * math.log(float(r)))/lamda


#Part - 2
dg = stats.dgamma(a=1)
norm = stats.norm(loc=0, scale=2)

x = np.linspace(min(dg.ppf(0.001), norm.ppf(0.001)),
                max(dg.ppf(0.999), norm.ppf(0.999)), 1000)
dg_samples = dg.pdf(x)
norm_samples = norm.pdf(x)

def rejection_sampling():
    while True:
        x = np.random.normal(0, 2)
        envelope = M * norm.pdf(x)
        p = np.random.uniform(0, envelope)
        if p < dg.pdf(x):
            return x

samples = [rejection_sampling() for x in range(10000)]

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