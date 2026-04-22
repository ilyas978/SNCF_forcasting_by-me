# Train Waiting Time Forecasting – Transilien SNCF Voyageurs

### 🛠 Feature Engineering & Importance
The model’s predictive power was driven by a robust feature set designed to capture the "pulse" of the SNCF network. I engineered **lag variables** to track historical delays, **cyclical temporal encodings** (sine/cosine transforms) for periodicity, and **station-level aggregations** to identify systemic bottlenecks. Using **Gain-based Feature Importance**, I verified that recent lag observations and station traffic density were the primary drivers of model accuracy, allowing for a refined, noise-reduced feature set.

### 📊 Model Comparison & Hyperparameter Tuning
To move beyond a baseline **Linear Regression (MAE: 0.84)**, I benchmarked several architectures, including Random Forest and XGBoost. **LightGBM** emerged as the superior model, offering the best balance of training speed and error reduction on the 667k-row dataset. I optimized the final model through a **Randomized Search** over the hyperparameter space—specifically tuning `num_leaves`, `feature_fraction`, and `learning_rate` with early stopping—ultimately achieving a **Top 10 ranking** with a final **MAE of 0.65** (a **~22% improvement**).
