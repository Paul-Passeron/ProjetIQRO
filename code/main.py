#!/usr/bin/env python3

import operator
import pprint
import time
from functools import reduce
from itertools import product
from typing import Generator

import numpy as np
import numpy.typing as npt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import HGate, MCXGate, QAOAAnsatz
from qiskit.primitives import StatevectorEstimator
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

mcx_gate = MCXGate(3)
hadamard_gate = HGate()

qc = QuantumCircuit(4)
qc.append(hadamard_gate, [0])
qc.append(mcx_gate, [0, 1, 2, 3])
qc.draw("mpl")


# Question 1
class MaxXorSat:
    def __init__(self, A: npt.NDArray[np.bool_], b: npt.NDArray[np.bool_]):
        self.m = A.shape[0]
        self.n = A.shape[1]
        assert b.shape[0] == self.m
        for x in A:
            assert x.shape[0] == self.n
        self.A = A
        self.b = b
        self._hamiltonian = None
        self._qaoa_ansatz = None

    # Question 2
    def solve_enumerate(self) -> tuple[list[bool], int]:
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
                return (list(map(lambda x: bool(x), res)), self.m)
            if current_fit > max_res:
                best_fit = res
                max_res = current_fit

        assert best_fit is not None
        return (list(map(lambda x: bool(x), best_fit)), max_res)

    # Question 2 (Version avec tous les résultats)
    def solve_all(self, min_level: int = 0) -> dict[int, tuple[npt.NDArray[np.bool_]]]:
        dict = {}
        for candidate in boolean_combinations(self.n):
            res = np.array(candidate)
            current_fit: int = 0
            for i, line in enumerate(self.A):
                log_and = np.logical_and(line, res)
                current_res: bool = False
                for x in log_and:
                    current_res = current_res != bool(x)
                current_fit += int(current_res == self.b[i])
                if current_fit >= min_level:
                    if current_fit not in dict.keys():
                        dict[current_fit] = [res]
                    else:
                        dict[current_fit].append(res)
        return dict

    # Question 4
    def polynome_bitstrings(self, xs: list[int]) -> int:
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

    def polynome(self, xs: list[int]) -> int:
        prod: int = reduce(operator.mul, map(lambda x: 1 - 2 * x, xs))
        return (1 - prod) // 2

    # Hamiltonien Hc
    def create_hamiltonian(self):
        if self._hamiltonian is None:
            pauli_list = []
            for j in range(self.m):
                one_indices = np.where(self.A[j, :] == 1)[0]
                pauli_str_list = ["I"] * self.n
                for idx in one_indices:
                    pauli_str_list[self.n - 1 - idx] = "Z"
                pauli_str = "".join(pauli_str_list)
                coeff = -0.5 * ((-1) ** (self.b[j]))
                pauli_list.append((pauli_str, coeff))
                id_str = "I" * self.n
                pauli_list.append((id_str, 0.5))
            self._hamiltonian = SparsePauliOp.from_list(pauli_list)

    def hamiltonian(self) -> SparsePauliOp:
        if self._hamiltonian is None:
            self.create_hamiltonian()
            assert self._hamiltonian is not None
        return self._hamiltonian

    def create_circuit(self, reps: int):
        if self._qaoa_ansatz is None:
            self._qaoa_ansatz = QAOAAnsatz(cost_operator=self.hamiltonian(), reps=reps)

    def qaoa_ansatz(self, reps: int) -> QAOAAnsatz:
        if self._qaoa_ansatz is None:
            self.create_circuit(reps)
            assert self._qaoa_ansatz is not None
        return self._qaoa_ansatz

    def solve_with_qaoa(self, reps: int) -> tuple[list[bool], int]:
        qaoa_ansatz = self.qaoa_ansatz(reps)
        num_params = qaoa_ansatz.num_parameters

        np.random.seed(8)
        init_params = 2 * np.pi * np.random.rand(num_params)

        pub = [qaoa_ansatz, [self.hamiltonian()], [init_params]]

        def cost_func(params):
            estimator = StatevectorEstimator()
            result = estimator.run(pubs=[pub]).result()  # pyright: ignore[reportArgumentType]
            cost = result[0].data.evs[0]  # pyright: ignore[reportAttributeAccessIssue]
            return cost

        best_params = minimize(cost_func, init_params, args=(), method="COBYLA")

        optimal_circuit = qaoa_ansatz.assign_parameters(best_params.x)
        assert optimal_circuit is not None

        final_state = Statevector(optimal_circuit)
        probs = final_state.probabilities_dict()

        # sort to have the best solution
        sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)

        best_sol, _ = sorted_probs[0]
        # qiskit inverts the order of qubits
        best_sol = best_sol[::-1]
        best_x = [bool(int(i)) for i in best_sol]
        score = 0
        for j in range(self.m):
            value = sum(self.A[j][k] * best_x[k] for k in range(self.n)) % 2
            if value == self.b[j]:
                score += 1
        return (best_x, score)
    
    # QUESTION 16 
    def solve_decision_with_grover(self, k:int):

        # QUESTION 14
        # on réutilise la fonction solve_all() de la question 2 avec k en argument
        def get_L():
            return list(self.solve_all(k).items())[0][1] # on prend uniquement le premier élément car il contient déjà toutes les solutions qui satisfont au moins k équations, incluant celle qui satisfait plus de k solutions.
        
        # Fonction qui contruit les petites portes de contrôle C_y pour chaque sol y de L comme décrit en question 13
        # On applique des X là ou il y a des 0 dans y et on applique une Z multicontrôlé
        def create_control_sol_gate(sol:list, idx_sol:int):
            qc = QuantumCircuit(self.n)
            
            for idx in range(len(sol)):
                if sol[idx]==0:
                    qc.x(self.n - 1 - idx) # qiskit inverse l'indice
            
            if self.n > 1:
                qc.mcp(np.pi, list(range(self.n-1)),self.n-1) # pour faire une multi control Z, on utilise mcp qui est une multi control phase gate avec un angle de pi qui correspond bien à une phase de -1.
            else:
                qc.z(0)

            # on "défait" les X
            for idx in range(len(sol)):
                if sol[idx]==0:
                    qc.x(self.n - 1 - idx) # qiskit inverse l'indice

            # print(qc)
            return qc.to_gate(label=f"C_y_{idx_sol}")
        
        def create_oracle_gate(L:list):
            qc = QuantumCircuit(self.n)
            for idx_sol in range(len(L)): # on applique les control de chaque solution successivement de sorte à prendre en compte toute les solutions de L dans l'oracle
                gate = create_control_sol_gate(L[idx_sol], idx_sol)
                qc.append(gate, list(range(0, self.n)))
            # print(qc)
            return qc.to_gate(label="Oracle")
        
        def create_symmetry_gate(): # symétrie autour de la moyenne 
            qc = QuantumCircuit(self.n)
            qc.h(range(self.n))
            qc.x(range(self.n))
            if self.n > 1:
                qc.mcp(np.pi, list(range(self.n-1)), self.n-1)
            else:
                qc.z(0)
            qc.x(range(self.n))
            qc.h(range(self.n))
            # print(qc)
            return qc.to_gate(label="Symmetry")
        
        
        L = get_L()
        if len(L)==0: # si classiquement on a pas de solution grover non plus ne trouvera pas de solution donc on peut déjà répondre "non" au problème de décision
            return (False, None)
        else:
            oracle_gate = create_oracle_gate(L)
            symmetry_gate = create_symmetry_gate()
            qr = QuantumRegister(self.n, 'q') # registre quantique
            cr = ClassicalRegister(self.n, 'c') # registre classique pour la mesure
            qc = QuantumCircuit(qr, cr)

            # On commence l'algorithme de Grover

            nbr_iter = int((np.pi/4)*np.sqrt(2**self.n)) # nombre optimal d'itération 

            # initialise l'état en une superposition uniforme avec une des H
            qc.h(qr)
            # oracle + symétrie nbr_iter fois
            for _ in range(nbr_iter):
                qc.append(oracle_gate, qr)
                qc.append(symmetry_gate, qr)
            
            #mesure
            qc.measure(qr, cr)
            # print(qc)

            sampler=Sampler()
            job = sampler.run([qc], shots=1000)
            data = job.result()[0].data
            counts = data.c.get_counts()

            # print(counts)

            sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)

            # print(sorted_counts[0][0])

            return (True, sorted_counts[0][0]) # return "oui" au problème de décision accompagné de la meilleure solution
    
    # QUESTION 18
    def solve_with_grover(self):
        k = self.m
        # on itère de m jusqu'à arriver à un k où on trouve au moins 1 solution
        result = self.solve_decision_with_grover(k)
        while(not(result[0])):
            k -= 1
            result = self.solve_decision_with_grover(k)
        best_x = [bool(int(i)) for i in result[1]]
        score = k
        return (best_x, score)


def is_odd(bitstring: list[int]) -> bool:
    res = 0
    for x in bitstring:
        res += x
    return res % 2 == 1


def boolean_combinations(n: int) -> Generator[list[int], None, None]:
    if n == 0:
        yield []
    else:
        for combo in boolean_combinations(n - 1):
            yield (combo + [0])
            yield (combo + [1])


# PARTIE 4 : EVALUATION DES ALGORITHMES

def random_max_xor_sat(n: int, m: int) -> MaxXorSat:
    random_A: npt.NDArray[np.bool_] = np.random.choice([True, False], size=(m, n))
    random_B: npt.NDArray[np.bool_] = np.random.choice([True, False], size=m)
    return MaxXorSat(random_A, random_B)


def eval_max_xor_sat(samples: int = 100, max_size: int = 10):
    from joblib import Parallel, delayed

    nms = product(*([range(2, max_size)] * 2))
    reps = 10
    d: dict[
        tuple[int, int], list[dict[str, dict[str, float | tuple[list[bool], int]]]]
    ] = {}

    def eval_single(nm):
        n, m = nm
        res = []
        for _ in range(samples):
            prob = random_max_xor_sat(n, m)
            results = {"qaoa": {}, "enumerate": {}}

            start = time.time()
            results["enumerate"]["solution"] = prob.solve_enumerate()
            end = time.time()
            results["enumerate"]["elapsed"] = end - start

            start = time.time()
            results["qaoa"]["solution"] = prob.solve_with_qaoa(reps)
            end = time.time()
            results["qaoa"]["elapsed"] = end - start

            res.append(results)
        return ((n, m), res)

    vals = Parallel(n_jobs=16)(delayed(eval_single)(nm) for nm in nms)

    for (n, m), results in vals:  # pyright: ignore[reportGeneralTypeIssues]
        d[(n, m)] = results

    return d


# Testing MaxXorSat
if __name__ == "__main__":
    # pprint.pp(eval_max_xor_sat(2, 4))


    A = np.array([[1, 1, 1, 0], [1, 0, 1, 0], [1, 0, 0, 0]])
    b = np.array([0, 1, 0])
    max_xor_sat = MaxXorSat(A, b)
    best_sol = max_xor_sat.solve_with_grover()
    print(best_sol)

    # x = max_xor_sat.solve_enumerate()

    # print(x)
    # p = max_xor_sat.polynome([1, 0, 1, 1])
    # print("Expect 1: ", p)
    # p = max_xor_sat.polynome([1, 1, 1, 1])
    # print("Expect 0: ", p)
    # p = max_xor_sat.polynome([0, 0, 1, 1])
    # print("Expect 0: ", p)
    # p = max_xor_sat.polynome([0, 0, 0, 1])
    # print("Expect 1: ", p)

    # max_xor_sat.create_hamiltonian()
    # print("Hamiltonien : \n", max_xor_sat.hamiltonian)

    # print(max_xor_sat.solve_with_qaoa(1))
