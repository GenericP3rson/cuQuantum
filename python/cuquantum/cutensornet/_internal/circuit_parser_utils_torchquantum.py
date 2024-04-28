# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import cupy as cp
import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import mat_dict
from torchquantum.util import switch_little_big_endian_matrix
from .tensor_wrapper import _get_backend_asarray_func

def remove_measurements(circuit):
    """
    Return a circuit with final measurement operations removed.
    """

    return circuit


def get_inverse_circuit(circuit):
    """
    Return a circuit with all gate operations inversed.
    """

    ops = circuit.op_history

    qmod = tq.QuantumModule.from_op_history(ops)
    qmod.inverse_module()
    qdev = tq.QuantumDevice(n_wires=circuit.n_wires, record_op=True)
    qmod(qdev)

    return qdev


def unfold_circuit(circuit, dtype='complex128', backend=cp):
    """
    Unfold the circuit to obtain the qubits and all gate tensors. 

    Args:
        circuit: A :class:`torchquantum.QuantumDevice` object. 
        dtype: Data type for the tensor operands.
        backend: The package the tensor operands belong to.

    Returns:
        All qubits and gate operations from the input circuit
    """
    asarray = _get_backend_asarray_func(backend)
    qubits = range(circuit.n_wires)
    gates = []
    old_gates = circuit.op_history
    for gate in old_gates:
        gate_qubits = gate["wires"]
        if type(gate_qubits) == int:
            gate_qubits = [gate_qubits]
        matrix = mat_dict[gate["name"]]
        if callable(matrix):
            matrix = matrix(torch.tensor([gate["params"]]))
        tensor = matrix.reshape((2,) * 2 * len(gate_qubits))
        tensor = asarray(tensor, dtype=dtype)
        gates.append((tensor, gate_qubits))
     
    return qubits, gates


def get_lightcone_circuit(circuit, coned_qubits):
    """
    Use unitary reversed lightcone cancellation technique to reduce the effective circuit size based on the qubits to be coned. 

    Args:
        circuit: A :class:`torchquantum.QuantumDevice` object. 
        coned_qubits: An iterable of qubits to be coned.

    Returns:
        A :class:`torchquantum.QuantumDevice` object that potentially contains less number of gates
    """
    coned_qubits = set(coned_qubits)
    all_operations = list(circuit.op_history)
    n_qubits = circuit.n_wires
    ix = len(all_operations)
    tail_operations = []
    while len(coned_qubits) != n_qubits and ix>0:
        ix -= 1
        operation = all_operations[ix]
        qubit_set = set(operation["wires"])
        if qubit_set & coned_qubits:
            tail_operations.append(operation)
            coned_qubits |= qubit_set
    ops = all_operations[:ix]+tail_operations[::-1]
    qmod = tq.QuantumModule.from_op_history(ops)
    qdev = tq.QuantumDevice(n_wires=n_qubits, record_op=True)
    qmod(qdev)
    return qdev
