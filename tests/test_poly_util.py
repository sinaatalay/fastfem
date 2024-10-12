import pytest
import numpy as np

from fastfem.elements import _poly_util


@pytest.fixture(scope="session")
def polynomials(request):
    class PolynomialFactory:
        def iterate():
            # null case
            yield []

            # monomials with coef 1
            for i in range(10):
                yield ([0] * i) + [1]

            # monomials with leading 0s
            for i in range(10):
                yield ([0] * i) + [1, 0, 0]

            # factors of x^2 - 1
            yield [-1, 1]
            yield [1, 1]

            yield [1, 0, 1]  # x^2 + 1 (irred)
            yield [1, 2, -3, -4]

            return [0]

        def __iter__(self):
            return PolynomialFactory.iterate()

    return PolynomialFactory()


def test_eval(polynomials):
    def eval_p(p, x):
        res = 0
        for i, c in enumerate(p):
            res += c * x**i
        return res

    for p in polynomials:
        for x in np.linspace(-20, 20, 300):
            px = eval_p(p, x)
            assert _poly_util.polyeval(p, x) == pytest.approx(px)


def test_deriv(polynomials):
    tol = 1e-5
    # we will use central finite difference which has O(h^2) error
    h = (tol * 1e-3) ** 0.5
    twoh = h * 2

    for p in polynomials:
        dp = _poly_util.polyderiv(p)
        for x in np.linspace(-20, 20, 300):
            cdiff = (
                _poly_util.polyeval(p, x + h) - _poly_util.polyeval(p, x - h)
            ) / twoh
            assert pytest.approx(_poly_util.polyeval(dp, x), rel=tol, abs=1e-9) == cdiff


def test_polyprod(polynomials):
    for p in polynomials:
        for q in polynomials:
            for x in np.linspace(-20, 20, 300):
                ref = pytest.approx(
                    _poly_util.polyeval(p, x) * _poly_util.polyeval(q, x)
                )
                assert (
                    _poly_util.polyeval(_poly_util._polyprod(p, q, clear_zeros=True), x)
                    == ref
                )
                assert (
                    _poly_util.polyeval(
                        _poly_util._polyprod(p, q, clear_zeros=False), x
                    )
                    == ref
                )


def test_integ(polynomials):
    def integ(p):
        # indefinite integral of p
        return [0] + [c / (i + 1) for i, c in enumerate(p)]

    for p in polynomials:
        intp = integ(p)
        # ensure our integral code is right
        dintp = _poly_util.polyderiv(intp)
        for x in np.linspace(-20, 20, 300):
            assert _poly_util.polyeval(p, x) == pytest.approx(
                _poly_util.polyeval(dintp, x)
            ), "test itself is a problem with integration (or deriv code is wrong)"
        assert _poly_util._polyinteg(p) == pytest.approx(
            _poly_util.polyeval(intp, 1) - _poly_util.polyeval(intp, -1)
        )


def test_polydot(polynomials):
    # assume integ and prod is correct
    for p in polynomials:
        for q in polynomials:
            assert _poly_util.polydot(p, q) == pytest.approx(
                _poly_util._polyinteg(_poly_util._polyprod(p, q))
            )


def test_leg_ortho():
    L = [_poly_util.leg(i) for i in range(8)]
    ONE = pytest.approx(1)
    ZERO = pytest.approx(0)
    for i in range(8):
        for j in range(8):
            if i == j:
                assert _poly_util.polydot(L[i], L[j]) == ONE
            else:
                assert _poly_util.polydot(L[i], L[j]) == ZERO
