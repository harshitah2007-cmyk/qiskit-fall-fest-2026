# ECG Biomedical Signal Classification with a Variational Quantum Classifier

## Project overview
This project explores binary classification of ECG beat samples using:
- A classical Logistic Regression baseline.
- A four-qubit Variational Quantum Classifier (VQC).
- Circuit compilation against supplied Processor A and Processor B connectivity configurations.

This is an educational prototype, not a clinically validated diagnostic tool.

## Workflow
1. Load prepared ECG training and test CSV files.
2. Use eight numeric features (`f1` to `f8`).
3. Standardize features and apply PCA with four components.
4. Scale the four components to the angle-encoding range [0, pi].
5. Train and evaluate Logistic Regression and VQC on the same balanced subset.
6. Train a four-qubit VQC using a ZZFeatureMap, RealAmplitudes ansatz, and COBYLA optimizer.
7. Evaluate accuracy, abnormal-class F1, and abnormal-class recall.
8. Transpile the quantum circuit against the supplied processor coupling maps and record circuit-resource metrics.

## Balanced evaluation results

The reported fair comparison uses 200 balanced training samples (100 per class) and 80 balanced test samples (40 per class), with random_state=42.

| Metric | Logistic Regression | VQC |
|---|---:|---:|
| Accuracy | 67.50% | 46.25% |
| Abnormal-class F1 | 62.86% | 35.82% |
| Abnormal-class recall | 55.00% | 30.00% |

On this small balanced test set, the classical baseline outperformed the VQC on all three reported metrics. These results should not be generalized beyond this experiment.

The earlier full-data Logistic Regression run achieved 74.66% accuracy, 4.55% abnormal-class F1, and 2.67% abnormal-class recall. It used a different training/test setup and must not be directly compared with the balanced-subset results above.

## Processor compilation results

| Metric | Processor A | Processor B |
|---|---:|---:|
| Physical qubits in supplied configuration | 5 | 9 |
| Logical qubits | 4 | 4 |
| Circuit depth | 23 | 23 |
| CX gates | 9 | 9 |
| SWAP gates | 0 | 0 |
| Total operations | 60 | 60 |

These are transpilation/resource metrics, not physical-hardware measurements or noise-aware performance results. Confirm Processor B's required qubit count against the organizer-approved Challenge Kit if the supplied materials appear inconsistent.

## Reproducibility
The notebook was developed in Google Colab. The environment used Qiskit 2.5.2 and Qiskit Machine Learning 0.9.1; Qiskit Algorithms was installed separately. Run the notebook cells in order and ensure the prepared dataset files are available at the paths expected by the notebook.

## Limitations
- The VQC experiment uses a small balanced subset.
- The classical and quantum models were compared on the same reported balanced test set, but this is a small experiment.
- No physical quantum hardware results or verified noise-aware results are claimed.
- Processor configurations should be checked against the official Challenge Kit.
- Performance is limited and must not be interpreted as clinical evidence.
