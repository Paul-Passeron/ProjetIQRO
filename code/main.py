import numpy as np
import numpy.typing as npt


class MaxOrSat:
    def __init__(self, A: npt.NDArray[np.bool_], b: npt.NDArray[np.bool_]):
        m = A.shape[0]
        n = A.shape[1]
        assert b.shape[0] == m
        for x in A:
            assert x.shape[0] == n
        self.A = A
        self.b = b


# Testing MaxOrSat
if __name__ == "__main__":
    A = np.array([[True, True], [False, True]])
    b = np.array([False, True])
    max_or_sat = MaxOrSat(A, b)
    # If no assert exception then it worked out
