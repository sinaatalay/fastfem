import numpy as np


# Bonnet's formula for finding Legendre polynomials
def leg(n):
    """Calculates the n^th degree legendre polynomial defined on [-1,1],
    scaled to have norm 1.
    The returned value is a coefficient array where the polynomial
    is given by sum(a[i] * x**i)
    """
    Pkm1 = np.array([1])  # P_{k-1} ; initially k=1
    Pk = np.array([0, 1])  # P_k     ; initially k=1
    if n == 0:
        return np.array([1]) * np.sqrt(0.5)
    if n == 1:
        return np.array([0, 1]) * np.sqrt(1.5)

    for k in range(1, n):
        Pkp1 = np.zeros(k + 2)  # P_{k+1} =
        Pkp1[1:] = (2 * k + 1) * Pk  # ( (2k+1) * x * P_k
        Pkp1[:-2] -= k * Pkm1  #  - n * P_{k-1}
        Pkp1 /= k + 1  # ) / (k+1)

        # inc k
        Pkm1 = Pk
        Pk = Pkp1

    return Pk * np.sqrt(n + 0.5)


def _polyinteg(p):
    """Integrates the polynomial p over the interval [-1,1]"""
    res = 0
    for i, a in enumerate(p):
        # integ a * x^i;
        if i % 2 == 0:
            # \int_0^1 x^i dx = 1/(i+1)
            res += 2 * a / (i + 1)
    return res


def _polyprod(p, q, clear_zeros=True):
    """Returns the polynomial product p*q.
    clear_zeros clears trailing (highest order) zero coefficients
    from p and q
    """
    if clear_zeros:
        while len(p) > 0 and p[-1] == 0:
            p = p[:-1]
        while len(q) > 0 and q[-1] == 0:
            q = q[:-1]
    lenp = len(p)
    lenq = len(q)
    if lenp == 0 or lenq == 0:
        return np.array([0])
    lenres = len(p) + len(q) - 1
    res = np.zeros(lenres)

    for i in range(lenres):
        # coefficient to x^i, as a sum of terms
        # p_0*q_i + p_1*q_{i-1} + ... + p_i*q_0
        #  = sum(p_j * q_{i-j})
        # start at i-j = min(i,deg(q)), end at j=min(i,deg(p))
        for j in range(max(0, i - lenq + 1), min(lenp, i + 1)):
            # print(i,j,res,p,q)
            res[i] += p[j] * q[i - j]
    return res


def polydot(p, q):
    """Returns the inner product of p and q over [-1,1]"""
    return _polyinteg(_polyprod(p, q))


def polyderiv(p):
    """Returns the derivative p'"""
    return np.array([a * (k + 1) for k, a in enumerate(p[1:])])


def polyeval(p, x):
    """Returns the evaluated p(x)"""
    return sum((a * x**i for i, a in enumerate(p)))


def get_GLL_knots(n):
    """Estimates the roots to be used for GLL quadrature"""
    roots = np.zeros(n + 1)
    # these ones are known
    roots[0] = -1
    roots[n] = 1

    # the rest are roots of P'; they are all separated and real.
    def find_roots(p, clear_zeros=True):
        # find roots of polynomial p in [-1,1] using bisection method
        # roots are assumed to be in between extreme values,
        # and we assume there are sufficient extreme values to sep zeros
        if clear_zeros:
            while p[-1] == 0:
                p = p[:-1]
        n = len(p) - 1
        if n == 0:
            return np.array([])
        # separators: extreme values
        seps = np.array([-1, *find_roots(polyderiv(p), False), 1])
        num_iters = int(np.ceil(-np.log2(1e-9)))
        # bisection
        a = seps[:-1]
        b = seps[1:]
        fa = polyeval(p, a)
        for _ in range(num_iters):
            c = (a + b) * 0.5
            fc = polyeval(p, c)
            goleft = fa * fc < 0
            a = np.where(goleft, a, c)  # a = a if goleft else c
            # similar to ^^, but we need to capture the possibility
            # that f(c) = 0
            b = np.where(goleft, c, np.where(fc == 0, c, b))
        return c

    roots[1:n] = find_roots(polyderiv(leg(n)))
    return roots


def build_GLL_polys(n):
    knots = get_GLL_knots(n)

    L = [[1]] * (n + 1)
    for i in range(n + 1):
        for j in range(n + 1):
            if i != j:
                L[i] = _polyprod(L[i], [-knots[j], 1]) / (knots[i] - knots[j])
    return L


def get_GLL_weights(n):
    knots = get_GLL_knots(n)
    P = leg(n)
    # the factor of (n+0.5) is undoing our normalization of P
    return (2 / (n * (n + 1))) * np.array(
        [1, *polyeval(P, knots[1:-1]) ** (-2) * (n + 0.5), 1]
    )
