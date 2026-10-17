"""Processor B topology configuration for the ECG VQC project.

This module records the supplied Processor B coupling map. It does not
claim access to physical quantum hardware or perform hardware execution.
"""
from qiskit.transpiler import CouplingMap

PROCESSOR_NAME = "B"
PHYSICAL_QUBITS = 9
COUPLING_EDGES = [
    (0, 1), (1, 2), (1, 6), (2, 3), (2, 6), (2, 5),
    (3, 4), (4, 5), (5, 6), (6, 7), (6, 8), (8, 0),
]


def build_coupling_map() -> CouplingMap:
    """Return Processor B's directed coupling map for transpilation."""
    return CouplingMap(couplinglist=COUPLING_EDGES)


if __name__ == "__main__":
    cmap = build_coupling_map()
    print(f"Processor {PROCESSOR_NAME}: {PHYSICAL_QUBITS} physical qubits")
    print("Coupling edges:", COUPLING_EDGES)
    print("Coupling map size:", cmap.size())
