# ECG Biomedical Signal Classification with a Variational Quantum Classifier

## Project overview

This project explores binary classification of ECG beat samples using:
- A classical Logistic Regression baseline.
- A four-qubit Variational Quantum Classifier (VQC).
- Circuit compilation against the supplied Processor A and Processor B
  connectivity configurations.

This is an educational prototype, not a clinically validated diagnostic tool.

## Workflow

1. Load the prepared ECG training and test CSV files.
2. Use eight numeric features (`f1` to `f8`).
3. Standardize features and apply PCA with four components.
4. Scale the four components to the angle-encoding range [0, pi].
5. Train a classical Logistic Regression baseline.
6. Train a four-qubit VQC using a ZZFeatureMap, RealAmplitudes ansatz,
   and COBYLA optimizer.
7. Evaluate accuracy, abnormal-class F1, and abnormal-class recall.
8. Transpile the quantum circuit against the supplied processor
   coupling maps and record circuit-resource metrics.

## Quantum experiment

The current VQC experiment uses 200 balanced training samples
(100 per class) and 80 balanced test samples (40 per class).

| Metric | VQC result |
|---|---:|
| Accuracy | 46.56% |
| Abnormal-class F1 | 32.82% |
| Abnormal-class recall | 30.00% |

These results are from a local quantum-circuit simulation.

## Processor compilation results

| Metric | Processor A | Processor B |
|---|---:|---:|
| Physical qubits in supplied configuration | 5 | 9 |
| Logical qubits | 4 | 4 |
| Circuit depth | 23 | 23 |
| CX gates | 9 | 9 |
| SWAP gates | 0 | 0 |
| Total operations | 60 | 60 |

These are transpilation metrics, not physical-hardware measurements.
Confirm Processor B's required qubit count against the organizer-approved
Challenge Kit: the supplied materials appear inconsistent.

## Classical comparison

The initial full-data Logistic Regression run achieved 74.66% accuracy,
4.55% abnormal-class F1, and 2.67% abnormal-class recall. It used a
different training/test setup, so these values must not be directly
compared with the VQC results.

Insert the Logistic Regression results from
`fair_model_comparison.csv` here before final submission.

## Project files

Expected files include:
- `main.ipynb` — complete runnable notebook.
- `ecg_train.csv` and `ecg_test.csv` — prepared dataset files.
- `baseline_results.csv` — initial classical baseline metrics.
- `vqc_results.csv` — VQC metrics.
- `fair_model_comparison.csv` — both models on the same test samples.
- `processor_A_metrics.csv` and `processor_B_metrics.csv`.
- `processor_comparison.csv` — processor resource comparison.
- `model_comparison.png` or `model_performance.png` — model performance plot.
- `processor_resources.png` — processor resource plot.
- `project_report.md` — experiment report.

## Reproducibility

The notebook was developed in Google Colab. The environment used
Qiskit 2.5.2 and Qiskit Machine Learning 0.9.1; the Qiskit Algorithms
package was installed separately. Exact package versions should be
recorded in the final environment specification.

Run the notebook cells in order. Ensure that the dataset files are
available at the paths expected by the notebook.

## Limitations

- The VQC experiment uses a small balanced subset.
- The full-data baseline and VQC do not share the same evaluation split.
- The fair comparison must be checked and included before submission.
- No physical quantum hardware results or verified noise-aware results
  are claimed.
- The official Processor B configuration must be confirmed.
- Performance is limited and must not be interpreted as clinical evidence.
