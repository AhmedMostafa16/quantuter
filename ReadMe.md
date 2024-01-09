# Quantum Circuit Simulator in Rust

A quantum circuit simulator in Rust. It executes instructions in Quanta, a custom yet simple language created for this project.

## Features

- Basic Gates
  - Pauli X
  - Pauli Y
  - Pauli Z
  - Hadamard
  - CNOT
  - Identity
  - Swap
  - Quantum Fourier Transform
- Measuring and gate execution circuits

# Quanta

## Initialization

Valid Quanta programs must commence with a circuit declaration. Declare a circuit using `circuit n 01011...`, where _n_ represents the number of qubits in the system, and `01011...` is an optional bitstring to initialize the state. If the bitstring is omitted, the state will default to |0>. For example:

```
; Create a circuit with 2 qubits, in the state |01>
circuit 2 01
```

## Comments

Introduce comments using `;`.

## Gates

Apply gates with `gate qubit1 qubit2... qubitn`, where 'gate' is the gate's name, and the parameters are integers indicating the targeted qubit(s).

The currently supported gates are as follows:

| Gate/Command       | Parameters | Description                     |
| ------------------ | ---------- | ------------------------------- |
| `identity`or`id`   | 1          | Does nothing to the qubit state |
| `h`                | 1          | Hadamard gate                   |
| `x`                | 1          | Pauli X gate (Not gate)         |
| `y`                | 1          | Pauli Y gate                    |
| `z`                | 1          | Pauli Z gate                    |
| `cnot`             | 2          | Conditional not gate            |
| `swap`             | 2          | Swap gate                       |
| `fourier` or `qft` | 0          | Qauntum Fourier Transform       |
| `measure`          | 1          | Measures the sellected qubit    |

## Example Quanta Programs

### Circuit Initial

```quanta
; Create a circuit with 2 qubits, in the state |01>
circuit 2 01
```

### Applying Gates

```qasm
; Initialize circuit
circuit 2

; Apply the Hadamard gate to the 0th qubit
h 0

; Apply the controlled-not gate with qubit 0 as the control and qubit 1 as the target
cnot 0 1
```

```qasm
; Deutsch-Jozsa Algorithm

; Initialize circuit with 2 qubits in the state |00>
circuit 2 00

; Oracle for the balanced function
; Apply X gate to all qubits for a balanced function
x 0
x 1

; Oracle for the constant function
; Do nothing for a constant function
;id 0
;id 1

; Apply Hadamard gate to the first qubit
h 1

; Measure the first qubit
measure 0

```

## Running the Simulator

The main Rust program reads a Quanta program from a file, tokenizes it, and simulates the quantum circuit based on the parsed instructions. The program supports various quantum gates and provides a way to measure the state or compute probabilities.

To run the simulator, provide the Quanta file as a command-line argument:

```bash
cargo run path/to/your/file.quanta
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
