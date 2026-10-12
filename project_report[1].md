# ECG Biomedical Signal Classification Using a Hybrid Quantum-Classical Model

## 1. Objective
The project explores binary classification of ECG beat samples into
normal and abnormal labels using a classical baseline and a hybrid
quantum-classical variational quantum classifier (VQC).

## 2. Dataset and preprocessing
The project uses the prepared ECG training and test CSV files.
Eight numeric features (f1-f8) are used as inputs. Standardization
and four-component Principal Component Analysis (PCA) are fitted
using training data only. The four PCA components are scaled to
[0, pi] for quantum angle encoding.

The quantum experiment uses a balanced subset of 200 training samples
(100 per class) and 80 test samples (40 per class).

## 3. Models
### Classical baseline
Logistic regression with standardized features and class weighting.

### Quantum model
A four-qubit VQC with a ZZFeatureMap, a RealAmplitudes ansatz,
a COBYLA optimizer, and a maximum of 30 optimizer iterations.
The VQC was evaluated using a local quantum-circuit simulation.

## 4. Quantum model results
- Accuracy: 0.4656 (46.56%)
- Abnormal-class F1: 0.3282 (32.82%)
- Abnormal-class recall: 0.3000 (30.00%)

The initial classical baseline on the full dataset achieved 74.66%
accuracy, 4.55% abnormal-class F1, and 2.67% abnormal-class recall.
However, that baseline used a different training/test setup, so it
must not be compared directly with the VQC results. Insert the
classical metrics from fair_model_comparison.csv before submission.

## 5. Processor compilation resource comparison
The same four-logical-qubit circuit was transpiled against the
supplied processor coupling maps.

| Metric | Processor A | Processor B |
|---|---:|---:|
| Physical qubits in supplied configuration | 5 | 9 |
| Logical qubits | 4 | 4 |
| Circuit depth | 23 | 23 |
| CX gates | 9 | 9 |
| SWAP gates | 0 | 0 |
| Total operations | 60 | 60 |

These are transpilation/resource metrics, not physical quantum
hardware measurements. Noise-aware results have not been established.

## 6. Discussion
The VQC achieved 30% recall for the abnormal-labelled class on the
small balanced test subset. Its observed accuracy was below 50%,
so performance remains limited. More experiments, repeated runs,
and feature/optimizer tuning would be required for a reliable
assessment.

## 7. Limitations
- The quantum evaluation uses a small balanced subset.
- The full-data classical baseline used a different split and cannot
  be used as a direct performance comparison.
- No physical quantum hardware execution is reported.
- Official noise settings have not been verified.
- The supplied documents appear inconsistent about Processor B's
  qubit count. The organizer-approved configuration must be checked.
- This is an educational prototype, not a clinically validated tool.

## 8. Reproducibility
The Colab notebook, dataset CSVs, model results, processor resource
tables, and generated graphs are saved in the project Google Drive
folder. Confirm that the notebook contains all code cells and that
the required files are included in the final repository.
