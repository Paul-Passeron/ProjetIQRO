import numpy as np
import numpy.typing as npt


class MaxXorSat:
    def __init__(self, A: npt.NDArray[np.bool_], b: npt.NDArray[np.bool_]):
        self.m = A.shape[0]
        self.n = A.shape[1]
        assert b.shape[0] == self.m
        for x in A:
            assert x.shape[0] == self.n
        self.A = A
        self.b = b

    def solve(self) -> tuple[npt.NDArray[np.bool_], int]:
        best_fit = None
        max_res = 0

        for candidate in boolean_combinations(self.n):
            res = np.array(candidate)
            current_fit: int = 0
            for i, line in enumerate(self.A):
                log_and = np.logical_and(line, res)
                current_res: bool = False
                for x in log_and:
                    current_res = current_res != x
                current_fit += int(current_res == self.b[i])
            if current_fit == self.m:
                return (res, self.m)
            if current_fit > max_res:
                best_fit = res
                max_res = current_fit

        assert best_fit is not None
        return (best_fit, max_res)

    def solve_all(self, min_level: int = 0) -> dict[int, tuple[npt.NDArray[np.bool_]]]:
        dict = {}
        for candidate in boolean_combinations(self.n):
            res = np.array(candidate)
            current_fit: int = 0
            for i, line in enumerate(self.A):
                log_and = np.logical_and(line, res)
                current_res: bool = False
                for x in log_and:
                    current_res = current_res != x
                current_fit += int(current_res == self.b[i])
                if current_fit >= min_level:
                    if current_fit not in dict.keys():
                        dict[current_fit] = [res]
                    else:
                        dict[current_fit].append(res)

        return dict

    def polynome(self, xs: list[int]) -> int:
        assert len(xs) == self.n
        res = 0
        bitstrings = boolean_combinations(self.n)
        for bitstring in bitstrings:
            if is_odd(bitstring):
                current_sum = 1
                for a, b in zip(bitstring, xs):
                    if a:
                        current_sum *= b
                    else:
                        current_sum *= 1 - b
                res += current_sum
        return res


def is_odd(bitstring: list[int]) -> bool:
    res = 0
    for x in bitstring:
        res += x
    return res % 2 == 1


def boolean_combinations(n: int) -> list[list[int]]:
    if n == 0:
        return [[]]
    res = []
    for combo in boolean_combinations(n - 1):
        res.append(combo + [0])
        res.append(combo + [1])
    return res


# Testing MaxOrSat
if __name__ == "__main__":
    A = np.array([[1, 1, 1, 0], [1, 0, 1, 0], [1, 0, 0, 0]])
    b = np.array([0, 1, 0])
    max_xor_sat = MaxXorSat(A, b)
    x = max_xor_sat.solve()
    d = max_xor_sat.solve_all()
    print(x)
    print(d)

    p = max_xor_sat.polynome([1, 0, 1, 1])
    print(p)
    p = max_xor_sat.polynome([1, 1, 1, 1])
    print(p)
    p = max_xor_sat.polynome([0, 0, 1, 1])
    print(p)
    p = max_xor_sat.polynome([0, 0, 0, 1])
    print(p)

    # If no assert exception then it worked out
