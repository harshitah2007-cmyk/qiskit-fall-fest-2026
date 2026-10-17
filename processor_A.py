"""Processor A topology configuration for the ECG VQC project.

This module records the supplied Processor A coupling map. It does not
claim access to physical quantum hardware or perform hardware execution.
"""
from qiskit.transpiler import CouplingMap

PROCESSOR_NAME = "A"
PHYSICAL_QUBITS = 5
COUPLING_EDGES = [(0, 1), (1, 2), (1, 3), (3, 4)]


def build_coupling_map() -> CouplingMap:
    """Return Processor A's directed coupling map for transpilation."""
    return CouplingMap(couplinglist=COUPLING_EDGES)


if __name__ == "__main__":
    cmap = build_coupling_map()
    print(f"Processor {PROCESSOR_NAME}: {PHYSICAL_QUBITS} physical qubits")
    print("Coupling edges:", COUPLING_EDGES)
    print("Coupling map size:", cmap.size())
