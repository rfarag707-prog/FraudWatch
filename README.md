# FraudWatch

A course project that prioritizes suspicious mobile-wallet transactions for review using classical machine learning on **PaySim simulation data**.

**Status:** implementation ready for local training. No PaySim performance or deployed app is claimed in this repository before the user runs training.

- Live application: add your deployed URL after testing it.
- Screenshot: add a screenshot from your trained application.
- Results: generated in `outputs/model_comparison.csv` and `outputs/report.json`.
- Arabic start guide: [START_HERE_AR.md](START_HERE_AR.md).

## Quick start
Python 3.11 or 3.12 recommended.
```bash
python -m venv .venv
# Activate your environment, or use its Python executable directly.
python -m pip install -r requirements.txt
python train.py --data data/paysim.csv --max-rows 300000
python -m streamlit run app.py
```
On Windows use the three numbered BAT files. The batch files use `.venv` directly.

## Experiment design
Uniform random-priority sampling reads the entire CSV in chunks and keeps at most the requested number of rows. Sampling does not depend on labels. Train, validation and test contain separate, chronologically ordered simulation steps (60%, 20%, 20% of unique steps, **not necessarily rows**).

A three-fold expanding-window cross-validation on training steps compares Logistic Regression, Random Forest and Histogram Gradient Boosting using average precision. Each candidate's imputer, scaler and encoder fit within the fold. A prior-probability DummyClassifier provides a baseline. The highest mean training CV average precision selects the model. A separate validation period selects a cost-minimizing threshold. Default costs (25 missed-fraud units versus 1 false-alarm unit) are explicit illustrative assumptions. The trained model is **not refitted** after threshold selection, so validation scores describe the saved model.

The frozen final test provides AP, ROC-AUC, precision, recall, F1 and confusion counts. All model-comparison rows use 0.5. The selected policy has its own threshold and metrics in report.json. Average precision is reported explicitly, rather than mislabeled trapezoidal PR-AUC. Test performance does not select the winner. Rerunning development after seeing test results requires a new independent test.

## Leakage and features
Inputs: step, type, amount, oldbalanceOrg, oldbalanceDest. Pure row-wise features derive hour, amount/origin-balance ratio and origin shortfall. Absolute step is used for splitting, while only hour is a model feature. Future balances, isFlaggedFraud, target and account identifiers are excluded. Labels must be complete. Missing balances use training-fold medians. Large amounts are retained because tails may carry fraud information; a derived ratio is clipped at 1e6 as an explicit numerical cap, not a learned outlier rule.

## Limitations
PaySim is a simulator, not an Egyptian wallet dataset. Scores are not calibrated probabilities. Uniform sampling can yield few positive examples and noisy estimates. The experiment reports point estimates, not confidence intervals. Temporal splitting does not establish unseen-customer generalization: repeated accounts can remain across periods even though IDs are excluded. Hypothetical costs are not real losses or savings. Feature associations do not prove causes of fraud. No production suitability claim is made.

## Outputs and delivery
The training command saves the fitted pipeline, configuration, model comparisons, training EDA, error analysis, HTML report, 12-slide PowerPoint/HTML presentation and a LinkedIn draft. Read the Arabic guide for deployment steps. The public application and repository still need publishing through your own accounts. Keep the raw dataset out of git. A successful code smoke test is not a model-performance result.

## Tests
```bash
python -m unittest discover -s tests -v
```
`tests/make_fixture.py` creates **artificial software-test rows**, not PaySim data. Never use that fixture or its scores as course evidence.

## Sources
- Dataset: https://www.kaggle.com/datasets/ealaxi/paysim1
- Pipeline: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
- Temporal validation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- AP definition: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
- Deployment: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

Course scope: supplied `ML_Final_Project_Ideas.pdf`, project 06. Add your name, actual contributions, measured results, app URL and screenshots before publishing.
