# ECG Biomedical Signal Classification Using a Hybrid Quantum-Classical Model

## 1. Objective
This project explores binary classification of ECG beat samples into normal and abnormal labels using a classical baseline and a hybrid quantum-classical variational quantum classifier (VQC).

## 2. Dataset and preprocessing
The project uses prepared ECG training and test CSV files. Eight numeric features (f1-f8) are used as inputs. Standardization and four-component Principal Component Analysis (PCA) are fitted using training data only. The four PCA components are scaled to [0, pi] for quantum angle encoding.

The reported balanced experiment uses 200 training samples (100 per class) and 80 test samples (40 per class).

## 3. Models
### Classical baseline
Logistic Regression, evaluated on the same balanced training and test subset as the VQC.

### Quantum model
A four-qubit VQC with a ZZFeatureMap, a RealAmplitudes ansatz, a COBYLA optimizer, and a maximum of 30 optimizer iterations. The VQC was evaluated using a local quantum-circuit simulation.

## 4. Balanced evaluation results

| Metric | Logistic Regression | VQC |
|---|---:|---:|
| Accuracy | 67.50% | 46.25% |
| Abnormal-class F1 | 62.86% | 35.82% |
| Abnormal-class recall | 55.00% | 30.00% |

On this small balanced test subset, Logistic Regression outperformed the VQC on all three reported metrics. The VQC's accuracy was below 50%, and its abnormal-class recall was 30%; these results indicate limited performance and should not be generalized beyond this experiment.

An earlier full-data Logistic Regression run achieved 74.66% accuracy, 4.55% abnormal-class F1, and 2.67% abnormal-class recall. That run used a different training/test setup and is not directly comparable to the balanced results above.

## 5. Processor compilation resource comparison
The same four-logical-qubit circuit was transpiled against the supplied processor coupling maps.

| Metric | Processor A | Processor B |
|---|---:|---:|
| Physical qubits in supplied configuration | 5 | 9 |
| Logical qubits | 4 | 4 |
| Circuit depth | 23 | 23 |
| CX gates | 9 | 9 |
| SWAP gates | 0 | 0 |
| Total operations | 60 | 60 |

These are transpilation/resource metrics, not physical quantum hardware measurements. Noise-aware results have not been established.

## 6. Discussion
The experiment shows that the tested classical baseline is stronger than this VQC configuration on the balanced subset. This is a useful negative result: the quantum classifier did not demonstrate an advantage in this run. Further work could explore optimizer settings, circuit depth, feature selection, larger repeated trials, and noise-aware evaluation, while preserving a fair shared split.

## 7. Limitations
- The quantum evaluation uses a small balanced subset.
- Results may vary with the sampler and optimization process.
- No physical quantum hardware execution or verified noise-aware performance is reported.
- Processor configurations should be confirmed against the organizer-approved Challenge Kit.
- This is an educational prototype, not a clinically validated tool.

## 8. Reproducibility
The Colab notebook and project CSVs/results are linked through the project repository and associated project folder. Run the notebook cells in order and ensure the prepared dataset files are available at the paths expected by the notebook.
