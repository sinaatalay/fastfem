import fastfem.fields.field as field
import numpy as np
import pytest


def shapes_generator(
    dimcountlims: tuple[int, int] = (0, 4),
    dimsizelims: tuple[int, int] = (0, 2),
    totalsizelims: tuple[int, int | None] = (0, 20),
):
    if (
        (dimcountlims[1] < dimcountlims[0] or dimcountlims[1] < 0)
        or (dimsizelims[1] < dimsizelims[0] or dimsizelims[1] < 0)
        or (
            totalsizelims[1] is not None and totalsizelims[1] < max(0, totalsizelims[0])
        )
    ):
        return
    if (dimcountlims[1] == 0) and (
        totalsizelims[0] <= 0 and (totalsizelims[1] is None or totalsizelims[1] >= 0)
    ):
        yield tuple()
        return

    yield from shapes_generator(
        (dimcountlims[0], dimcountlims[1] - 1), dimsizelims, totalsizelims
    )
    for shape in shapes_generator(
        (dimcountlims[1] - 1, dimcountlims[1] - 1), dimsizelims, totalsizelims
    ):
        for k in range(dimsizelims[0], dimsizelims[1] + 1):
            newshape = shape + (k,)
            size = np.prod(newshape, dtype=int)
            if totalsizelims[0] <= size and (
                totalsizelims[1] == None or size <= totalsizelims[1]
            ):
                yield newshape


def shapes():
    yield from shapes_generator(dimcountlims=(0, 3))
    yield from ((1,) * i + (3, 2) for i in range(3, 5))
    yield from ((1,) * i + (3, 2, 2) for i in range(3, 5))


def shapes_small():
    # yield from shapes_generator(dimcountlims=(0,2),dimsizelims=(1,2))
    # yield from ((1,)*i + (3,2) for i in range(3,5))
    # yield from ((1,)*i + (3,2,2) for i in range(3,5))
    # yield from shapes_generator(dimcountlims=(1,2),dimsizelims=(0,1))
    yield tuple()
    yield from ((i,) for i in range(3))
    yield from ((1,) * i + (3, 5) for i in range(2))
    yield from ((1,) * i + (5, 3) for i in range(2))
    yield (2, 1, 2)
    yield (0, 1)


def test_shape_broadcastability_and_compatibility():
    for a in shapes():
        for b in shapes():
            for c in shapes():
                broadcastable = field._is_broadcastable(a, b, c)
                compatible = field._is_compatible(a, b, c)
                try:
                    target = np.broadcast_shapes(a, b, c)
                    assert compatible, (
                        f"Numpy broadcasts {a}, {b}, {c} -> {target}, but"
                        f" _is_compatible(...) == {compatible}, which should be True."
                    )
                    assert (target == a) == broadcastable, (
                        f"Numpy broadcasts {a}, {b}, {c} -> {target}, but"
                        f" _is_broadcastable(...) == {compatible}, which should be"
                        f" {target == a}."
                    )
                except ValueError:
                    assert not (broadcastable or compatible), (
                        f"Numpy failed to broadcast {a}, {b}, {c}, but this disagrees"
                        " with field functions: _is_broadcastable(...) =="
                        f" {broadcastable}, _is_compatible(...) == {compatible}."
                    )


def test_field_broadcast_compatibility_on_triples():
    for a in shapes_small():
        for b in shapes_small():
            for c in shapes_small():
                for a_ in shapes_small():
                    for b_ in shapes_small():
                        for c_ in shapes_small():
                            nonconst = [
                                s for s in (a, b, c) if np.prod(s, dtype=int) != 1
                            ]
                            compatibility = field._is_compatible(a_, b_, c_) and (
                                len(nonconst) == 0
                                or all(nonconst[0] == s for s in nonconst)
                            )
                            fa = field.Field(a, tuple(), np.empty(a + a_))
                            fb = field.Field(b, tuple(), np.empty(b + b_))
                            fc = field.Field(c, tuple(), np.empty(c + c_))
                            assert (
                                field.Field.are_compatible(fa, fb, fc) == compatibility
                            ), (
                                f"{fa.shape}, {fb.shape} and"
                                f" {fc.shape} broadcastibility should be"
                                f" {compatibility}."
                            )
                            if compatibility:
                                fa2, fb2, fc2 = (
                                    field.Field.broadcast_field_compatibility(a, b, c)
                                )
                                assert fa == fa2
                                assert fb == fb2
                                assert fc == fc2


def test_field_broadcast_full_on_doubles():
    for a in shapes_small():
        for b in shapes_small():
            for a_ in shapes_small():
                for b_ in shapes_small():
                    for a__ in shapes_small():
                        for b__ in shapes_small():
                            broadcastibility = (
                                field._is_compatible(a_, b_)
                                and field._is_compatible(a__, b__)
                                and (
                                    np.prod(b, dtype=int) == 1
                                    or np.prod(a, dtype=int) == 1
                                    or a == b
                                )
                            )
                            fa = field.Field(a, a__, np.empty(a + a_ + a__))
                            fb = field.Field(b, b__, np.empty(b + b_ + b__))
                            assert (
                                field.Field.are_broadcastable(fa, fb)
                                == broadcastibility
                            ), (
                                f"{fa.shape} and {fb.shape} broadcastibility should be"
                                f" {broadcastibility}."
                            )
                            if broadcastibility:
                                fa2, fb2 = field.Field.broadcast_fields_full(fa, fb)
                                assert fa == fa2
                                assert fb == fb2
