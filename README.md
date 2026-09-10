# Machine Learning for Top-Jet Classification and Jet Energy Reconstruction

A particle-physics machine-learning project for distinguishing **boosted Top jets from QCD background jets** using simulated collider events.

The project combines exploratory particle-physics data analysis, feature engineering, neural-network classification, uncertainty estimation, physics-driven event selection, data augmentation, hyperparameter optimization, and jet-energy regression.

## Overview

Separating jets originating from heavy-particle decays from ordinary QCD jets is an important classification problem in collider physics.

In this project, a neural network is trained to distinguish:

* **QCD jets:** label 0
* **Top jets:** label 1

Rather than considering only overall classification accuracy, the analysis also addresses an experimentally relevant operating condition:

> Keep the fraction of QCD jets incorrectly selected as Top below **5%**, while retaining as many genuine Top jets as possible.

The project therefore combines conventional machine-learning metrics with threshold optimization and uncertainty estimation.

## Dataset

The dataset consists of simulated collider jet events separated into predefined training, validation, and test sets.

| Split      | Events |
| ---------- | -----: |
| Training   |  7,606 |
| Validation |  2,516 |
| Test       |  4,039 |

The training and validation datasets are imbalanced:

| Split      |   QCD |   Top |
| ---------- | ----: | ----: |
| Training   | 6,106 | 1,500 |
| Validation | 2,012 |   504 |
| Test       | 2,034 | 2,005 |

The training sample therefore contains approximately **80% QCD and 20% Top events**, while the test set is approximately balanced.

No missing values were found in the supplied datasets.

## Input Variables

Six jet-level variables are initially available:

* `jet_pt`
* `jet_eta`
* `jet_phi`
* `jet_energy`
* `jet_mass`
* `jet_nparticles`

These represent jet kinematics, direction and basic information about the internal jet structure.

## Exploratory Data Analysis

The first stage investigates the distributions and discriminative power of the input variables using:

* class-dependent feature distributions
* Spearman correlation
* univariate ROC-AUC
* PCA
* UMAP
* feature-range and outlier checks

### Main Observation

The strongest individual discriminator is **jet mass**.

Its univariate ROC-AUC is approximately:

```text
jet_mass ≈ 0.903
```

Particle multiplicity is the second most informative individual variable:

```text
jet_nparticles ≈ 0.710
```

In contrast, `jet_pt`, `jet_eta`, `jet_phi`, and `jet_energy` individually provide relatively little class separation.

The Spearman analysis gives the same qualitative picture:

```text
jet_mass          ρ ≈ 0.555
jet_nparticles    ρ ≈ 0.289
```

PCA also indicates that the largest class-related variation is associated with a component dominated by **jet mass and particle multiplicity**.

## Physics-Inspired Feature Engineering

The raw features are supplemented with several derived quantities.

### Angular Encoding

Because azimuthal angle φ is periodic, raw `jet_phi` is replaced by:

```python
sin_phi = sin(jet_phi)
cos_phi = cos(jet_phi)
```

This avoids the artificial discontinuity between \(-\pi\) and \(+\pi\).

### Dimensionless Ratios

Two additional kinematic relationships are constructed:

```text
mass_over_pt
energy_over_pt
```

The mass-to-transverse-momentum ratio provides slightly stronger standalone separation than jet mass alone, with a univariate ROC-AUC of approximately **0.904**.

### Final Classification Features

The neural network uses nine inputs:

```text
jet_pt
jet_eta
jet_energy
jet_mass
jet_nparticles
sin_phi
cos_phi
mass_over_pt
energy_over_pt
```

All inputs are standardized using statistics calculated **only from the training dataset**.

## Neural Network Classifier

The baseline classifier is a fully connected multilayer perceptron implemented in **PyTorch**.

Architecture:

```text
9 input features
       │
       ▼
Dense(64)
       │
      ReLU
       │
   Dropout(0.2)
       │
       ▼
Dense(16)
       │
      ReLU
       │
   Dropout(0.2)
       │
       ▼
Dense(1)
       │
       ▼
Top probability
```

Binary classification is trained with `BCEWithLogitsLoss`.

Because Top jets are underrepresented in the training data, the positive class is weighted according to the training class ratio:

```text
positive-class weight ≈ 4.07
```

The model is optimized with Adam and early stopping based on validation loss.

## Baseline Classification Results

On the held-out test set, the baseline model achieves:

| Metric                  |     Result |
| ----------------------- | ---------: |
| Accuracy                | **0.8950** |
| ROC-AUC                 | **0.9488** |
| PR-AUC — Top            | **0.9205** |
| PR-AUC — QCD            | **0.9625** |
| Macro Average Precision | **0.9415** |

At the default probability threshold of 0.5, the confusion matrix is:

```text
                 Predicted
              QCD       Top

True QCD      1707       327
True Top        97      1908
```

This operating point provides very high Top recall, but the number of QCD jets misidentified as Top is too large for the stricter physics-selection requirement considered later.

## Hyperparameter Optimization

The neural-network hyperparameters were also optimized using **Optuna**.

The optimized model produced approximately:

| Metric   | Test Result |
| -------- | ----------: |
| Accuracy |   **0.895** |
| ROC-AUC  |  **0.9496** |
| PR-AUC   |   **0.927** |

The largest improvement relative to the original baseline is observed in PR-AUC.

This demonstrates that tuning improves ranking performance, although the baseline network already provides strong discrimination.

## Epistemic Uncertainty

Classification performance alone does not indicate whether the model knows when its prediction is unreliable.

To investigate this, **deep ensembles** are used to estimate epistemic uncertainty.

Several independently trained neural networks are used to predict the same event. Their mean probability provides the ensemble prediction, while disagreement between the models represents epistemic uncertainty.

For each event:

```text
Ensemble models
      │
      ├── p₁(Top)
      ├── p₂(Top)
      ├── p₃(Top)
      └── ...
      │
      ▼
Mean prediction + prediction variance
```

Both ensemble variance and mutual-information-based quantities are investigated.

### Ensemble Classification

The ensemble achieves approximately:

| Metric   |     Result |
| -------- | ---------: |
| Accuracy | **0.8943** |
| ROC-AUC  | **0.9460** |
| PR-AUC   | **0.9173** |

The purpose of the ensemble is therefore not primarily to increase classification accuracy, but to provide information about **prediction reliability**.

## Does Uncertainty Identify Model Errors?

Yes.

Incorrectly classified events tend to have greater ensemble disagreement than correctly classified events.

Using epistemic uncertainty alone to distinguish **wrong predictions from correct predictions** gives:

```text
AUROC for error detection ≈ 0.815
```

This indicates that uncertainty contains useful information about model failures.

## Selective Prediction

The model can also abstain from making predictions on highly uncertain events.

When events are ranked by uncertainty and the least certain examples are rejected, classification accuracy on the retained events increases.

The notebook shows approximately:

```text
100% coverage → accuracy ≈ 89.5%
 50% coverage → accuracy ≈ 98%
```

This demonstrates a practical role for uncertainty estimation: uncertain events can be flagged for additional analysis instead of treating every model prediction as equally reliable.

## Physics-Motivated Event Selection

For this experiment, the important requirement is:

```text
QCD false-positive rate ≤ 5%
```

In other words, no more than 5% of genuine QCD jets should be incorrectly accepted as Top jets.

The probability threshold is selected **only using the validation dataset**. Among thresholds satisfying the 5% QCD false-positive constraint, the threshold retaining the largest fraction of Top jets is chosen.

The resulting threshold is:

```text
Top probability threshold ≈ 0.863
```

On validation data:

```text
QCD FPR = 4.92%
Top efficiency = 61.71%
```

The threshold is then frozen and evaluated on the independent test set.

### Test-Set Operating Point

```text
QCD FPR        = 3.88%
Top efficiency = 59.50%
```

The corresponding test confusion matrix is:

```text
TN = 1955
FP =   79
FN =  812
TP = 1193
```

Therefore, the required condition is successfully satisfied:

```text
3.88% QCD false-positive rate < 5%
```

while approximately **59.5% of Top jets are retained**.

This operating-point analysis is more relevant to the experimental objective than simply maximizing accuracy at a threshold of 0.5.

## Physics-Motivated Data Augmentation

Data augmentation was investigated as a method for increasing effective training diversity while preserving physically reasonable jet properties.

Augmentation is applied **only to the training data**. Validation and test samples remain unchanged so that model comparisons remain fair.

The augmented model obtains:

| Model     |   Accuracy | ROC-AUC |     PR-AUC |
| --------- | ---------: | ------: | ---------: |
| Baseline  |     0.8950 |  0.9488 |     0.9205 |
| Augmented | **0.8975** |  0.9486 | **0.9214** |

The overall improvement is modest.

However, at the physics-motivated operating point requiring QCD FPR ≤ 5%, the augmented model achieves:

```text
QCD FPR        = 4.52%
Top efficiency = 59.70%
```

compared with:

```text
QCD FPR        = 3.88%
Top efficiency = 59.50%
```

for the baseline.

Thus, the augmentation experiment produces a small increase in retained Top events while remaining within the allowed background constraint.

## Model Comparison

The main classification experiments can be summarized as:

| Model            |   Accuracy |     ROC-AUC |      PR-AUC |
| ---------------- | ---------: | ----------: | ----------: |
| Baseline MLP     |     0.8950 |      0.9488 |      0.9205 |
| Augmented MLP    | **0.8975** |      0.9486 |      0.9214 |
| Optuna-tuned MLP |     ≈0.895 | **≈0.9496** | **≈0.9270** |

The differences are relatively small because the baseline classifier already provides strong discrimination.

The Optuna-tuned model gives the strongest ranking metrics, particularly PR-AUC.

## Jet Energy Regression

As an additional study, the project investigates whether jet energy can be reconstructed from the remaining jet observables after selecting Top events.

Only Top jets are retained for this experiment.

The regression model uses:

```text
jet_pt
jet_eta
jet_phi
jet_mass
jet_nparticles
```

to predict:

```text
jet_energy
```

The regression network has the architecture:

```text
5 inputs → 64 → 16 → 1
```

and is trained using mean squared error with AdamW optimization and validation-based early stopping.

## Energy Regression Results

On the Top-jet test sample, the neural-network regressor achieves:

| Metric                       |        Result |
| ---------------------------- | ------------: |
| MAE                          |  **8.07 GeV** |
| RMSE                         | **13.17 GeV** |
| R²                           |   **0.99829** |
| Mean relative absolute error |    **0.878%** |
| Predictions within ±10 GeV   |     **74.5%** |
| Predictions within ±20 GeV   |     **93.5%** |
| Predictions within 5%        |    **99.65%** |
| Predictions within 10%       |      **100%** |

Because this result is unusually strong, additional sanity checks were performed.

## Regression Sanity Checks

### Linear Baseline

A simple linear model using `jet_pt` and `jet_eta` performs poorly:

```text
R² ≈ 0.0059
MAE ≈ 243 GeV
```

Using all available regression inputs in ordinary linear regression still gives only:

```text
R² ≈ 0.0086
MAE ≈ 243 GeV
```

### Shuffled-Target Test

Randomly shuffling the target removes the relationship between input variables and energy and produces a negative \(R^2\), as expected.

### Duplicate Check

No exact duplicate feature rows were found between the Top-jet training and test datasets.

### Physics Cross-Check

The unusually accurate energy reconstruction was ultimately explained by the underlying jet kinematics.

Using the standard relationship between transverse momentum, pseudorapidity, mass and energy reconstructs the supplied jet energy essentially exactly:

```text
R² ≈ 1.0
MAE ≈ 0.00004 GeV
```

This is an important result of the project: although the neural network can learn the nonlinear mapping extremely accurately, the analytical physics relationship remains the appropriate estimator when it is known.

It also illustrates why high ML performance should always be subjected to **physics-based sanity checks rather than accepted solely from numerical metrics**.

## Main Results

The project demonstrates several complementary aspects of machine learning for particle physics:

```text
Top/QCD discrimination
        │
        ├── ROC-AUC ≈ 0.95
        │
        ├── QCD FPR constrained below 5%
        │
        └── ~59.5% Top efficiency
        │
        ▼
Epistemic uncertainty
        │
        ├── Deep ensembles
        ├── Error-detection AUROC ≈ 0.815
        └── Selective prediction
        │
        ▼
Model optimization
        │
        ├── Physics-inspired features
        ├── Data augmentation
        └── Optuna tuning
        │
        ▼
Jet-energy regression
        │
        └── R² ≈ 0.998
```

## Technologies

* Python
* PyTorch
* NumPy
* pandas
* scikit-learn
* Matplotlib
* UMAP
* Optuna
* Jupyter Notebook

## Running the Project

The notebook expects the supplied HDF5 dataset:

```text
data.h5
```

with three keys:

```text
train
val
test
```

The dataset itself is not included in this repository unless its distribution terms explicitly permit redistribution.

Install the main dependencies with:

```bash
pip install numpy pandas matplotlib scikit-learn torch optuna umap-learn tables
```

Then update the dataset path in the notebook if necessary and run the cells sequentially.

## Project Scope

This is an **academic machine-learning study using simulated collider jet events**.

The results should therefore be interpreted as performance on the provided simulated dataset rather than as the performance of a production classifier deployed in an experimental collider analysis.

The project focuses on demonstrating a complete and critically evaluated particle-physics ML workflow:

* understanding the physics variables before modelling
* constructing physically motivated features
* accounting for class imbalance
* training and optimizing neural networks
* evaluating performance beyond simple accuracy
* satisfying a physics-driven background-rejection requirement
* quantifying epistemic uncertainty
* testing uncertainty as an indicator of model failure
* investigating physically meaningful augmentation
* validating unexpectedly strong ML results against known physics

## Key Takeaway

The final classifier achieves approximately **0.95 ROC-AUC** for Top-vs-QCD discrimination and can be operated at a point where only **3.88% of QCD jets are misidentified as Top while retaining 59.5% of Top jets**.

Beyond classification performance, the project demonstrates that **deep-ensemble uncertainty can identify unreliable predictions**, achieving an error-detection AUROC of approximately **0.815**.

The energy-regression study further highlights an important principle in scientific machine learning: strong predictive performance should always be checked against the underlying physics, particularly when an analytical relationship may already explain the target.
