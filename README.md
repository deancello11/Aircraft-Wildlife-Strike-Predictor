# Wildlife Strike Damage Prediction

Predicts whether an aircraft wildlife strike caused damage, using records from the FAA Wildlife Strike Database. The model is an ensemble of two gradient boosting classifiers with a decision threshold tuned for balanced accuracy.

## Problem

Given details of a bird or wildlife strike (aircraft, species, phase of flight, airport, time, and more), predict the binary target `INDICATED_DAMAGE`.

- **Training data:** 307,178 records, 55 columns
- **Test data:** 34,131 records
- **Class balance:** about 6.4% of strikes involve damage, so plain accuracy would be misleading. The metric is **balanced accuracy**.

## Approach

**Preprocessing and features**
- Dropped identifiers, free text, and mostly empty columns (for example `REG`, `LOCATION`, `BIRD_BAND_NUMBER`).
- Date features: day of week, day of year, quarter, weekend flag.
- Parsed time of day into minutes and hour, with a flag for unknown times.
- Converted bucketed counts (`NUM_SEEN`, `NUM_STRUCK`) to numeric midpoints, and mapped bird size to an ordinal scale.
- Added missing-value indicators for each column, plus text length, word count, and a damage-keyword flag from the comments.
- Median imputation for numeric columns and ordinal encoding for categorical columns (rare categories grouped, unseen categories handled).

**Model**
- Two `HistGradientBoostingClassifier` models with different settings, with predicted probabilities averaged:
  - Slower learning rate (0.04), 31 leaf nodes, L2 = 0.05
  - Faster learning rate (0.06), 127 leaf nodes, min 30 samples per leaf, L2 = 0.01
- Balanced class weights and early stopping on both.
- Decision threshold chosen on the validation set to maximize balanced accuracy (best value 0.46).
- Final models retrained on the full training set to generate `submission.csv`.

## Results

Evaluated on a stratified 80/20 train/validation split.

| Metric | Value |
|---|---|
| Balanced accuracy (threshold 0.50) | 0.8639 |
| Balanced accuracy (tuned threshold 0.46) | **0.8648** |
| Damage class recall | 0.846 |
| Damage class precision | 0.331 |
| Overall accuracy | 0.881 |

The low precision on the damage class is expected given the class imbalance: the model catches most damaging strikes at the cost of more false alarms.

## Notes and limitations

- The threshold was tuned on the same validation split used to report the score, so the figure is slightly optimistic.
- Some features come from comment text and update dates, which are recorded after an incident and may carry information about the outcome. They would need review before any real-time use.
- No single-model baseline is included, so the benefit of the ensemble is not measured here.

## Run it

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

1. Download the data from [source link] and place `train.csv`, `test.csv`, and `sample_submission.csv` at the paths set at the top of the notebook.
2. Open `[notebook-name].ipynb` and run all cells.
3. The notebook writes `submission.csv`.

The data files are not included in this repository.

## Tools

Python 3.11, pandas, NumPy, scikit-learn, matplotlib, seaborn

## Author

Dean Chaudhry
