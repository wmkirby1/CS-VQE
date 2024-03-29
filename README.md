# CS-VQE Ansatz Builder

Based on the foundational CS-VQE work of [https://quantum-journal.org/papers/q-2021-05-14-456/](https://quantum-journal.org/papers/q-2021-05-14-456/), making use of the original code found at [https://github.com/wmkirby1/ContextualSubspaceVQE](https://github.com/wmkirby1/ContextualSubspaceVQE).

## What is included

- All of the original functionality of [ContextualSubspaceVQE](https://github.com/wmkirby1/ContextualSubspaceVQE);
- Qubit tapering funtionality making use of [Qiskit Nature](https://qiskit.org/documentation/nature/);
- Qubit conversion between Qiskit, OpenFermion and AQASM representations;
- Circuit class for constructing and executing CS-VQE Ansatz circuits;
- An implementation of the *noncontextual projection Ansatz*

## How to cite

When you use ContextualSubspaceVQE in a publication or other work, please cite as:

> Kirby, William M., Andrew Tranter, and Peter J. Love. "Contextual Subspace Variational Quantum Eigensolver." Quantum 5 (2021): 456., [DOI: 10.22331/q-2021-05-14-456](https://quantum-journal.org/papers/q-2021-05-14-456/).

## How to use

Refer to [usage.ipynb](https://github.com/TimWeaving/CS-VQE/blob/main/usage.ipynb) for an overview of the basic functionality.
