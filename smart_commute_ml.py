"""
Smart Commute ML: Travel Time Prediction and Transport Recommendation
MDM-331-ETC Machine Learning Lab, Experiment 10 (Mini Project)
Author: Shrawani Thorat

HOW TO USE WITH YOUR REAL DATA
  1. Collect responses (Google Form) with the same columns as COLUMNS below.
  2. Save as commute_dataset.csv next to this file.
  3. Run: python smart_commute_ml.py
If commute_dataset.csv is missing, a SIMULATED dataset is generated so the pipeline can run.
"""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
OUT = "outputs"; os.makedirs(OUT, exist_ok=True)
SEED = 42
rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------ 1. DATA
COLUMNS = ["Start_Area", "Destination", "Distance_km", "Day", "Reach_Hour",
           "Weather", "Traffic_Level", "Transport_Mode", "Cost_Rs", "Travel_Time_min"]

def simulate_dataset(n=600):
    """SIMULATED commute records for Pune. Replace with real survey data."""
    base = { # approx road km from area to destination
        "PVG College": dict(Katraj=6, Swargate=9, Bibwewadi=7, Hadapsar=16, Wakad=22, Kothrud=14, Shivajinagar=13),
        "COEP": dict(Katraj=11, Swargate=5, Bibwewadi=8, Hadapsar=10, Wakad=14, Kothrud=6, Shivajinagar=2),
        "MIT Pune": dict(Katraj=14, Swargate=10, Bibwewadi=12, Hadapsar=9, Wakad=17, Kothrud=3, Shivajinagar=7),
        "Infosys": dict(Katraj=22, Swargate=19, Bibwewadi=20, Hadapsar=14, Wakad=5, Kothrud=12, Shivajinagar=14),
        "TCS": dict(Katraj=21, Swargate=17, Bibwewadi=19, Hadapsar=8, Wakad=15, Kothrud=13, Shivajinagar=12),
        "Wipro": dict(Katraj=23, Swargate=18, Bibwewadi=21, Hadapsar=7, Wakad=16, Kothrud=14, Shivajinagar=13),
    }
    speed = {"Bike": 27, "Car": 23, "Bus": 16, "Shared Auto": 19}
    wait = {"Bike": 0, "Car": 0, "Bus": 8, "Shared Auto": 5}
    cost_km = {"Bike": 2.5, "Car": 8.0, "Bus": 1.8, "Shared Auto": 4.0}
    tmul = {"Low": 1.0, "Medium": 1.3, "High": 1.7}
    rows = []
    for _ in range(n):
        dest = rng.choice(list(base)); area = rng.choice(list(base[dest]))
        dist = max(1.0, base[dest][area] * rng.uniform(0.9, 1.15))
        day = rng.choice(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"], p=[.2, .18, .18, .18, .18, .08])
        hour = rng.choice([8, 8.5, 9, 9.5, 10, 10.5], p=[.2, .2, .22, .16, .14, .08])
        weather = rng.choice(["Clear", "Cloudy", "Rain"], p=[.5, .25, .25])
        peak = 1.15 if 8.5 <= hour <= 9.5 else 1.0
        traffic = rng.choice(["Low", "Medium", "High"], p=[.25, .4, .35] if peak > 1 else [.45, .4, .15])
        mode = rng.choice(list(speed), p=[.35, .15, .3, .2])
        t = dist / speed[mode] * 60 * tmul[traffic] * peak * (1.2 if weather == "Rain" else 1.0)
        t += wait[mode] * rng.uniform(.5, 1.5) + rng.normal(0, 3)
        cost = dist * cost_km[mode] * rng.uniform(.9, 1.1)
        rows.append([area, dest, round(dist, 1), day, hour, weather, traffic, mode, round(cost, 1), round(max(t, 4), 1)])
    df = pd.DataFrame(rows, columns=COLUMNS)
    # inject realistic data-quality issues so cleaning is meaningful
    for c, k in [("Traffic_Level", 12), ("Weather", 8), ("Distance_km", 10)]:
        df.loc[rng.choice(df.index, k, replace=False), c] = np.nan
    df.loc[rng.choice(df.index, 3, replace=False), "Travel_Time_min"] = [-5, 400, 350] # invalid entries
    df = pd.concat([df, df.sample(8, random_state=1)], ignore_index=True)           # duplicates
    return df

SIMULATED = not os.path.exists("commute_dataset.csv")
if SIMULATED:
    df = simulate_dataset(); df.to_csv("commute_dataset.csv", index=False)
else:
    df = pd.read_csv("commute_dataset.csv")
print("DATA SOURCE:", "SIMULATED" if SIMULATED else "REAL SURVEY")

# ------------------------------------------------------------------ 2. UNDERSTAND
print(df.shape); print(df.dtypes); print(df.describe(include="all").T)
quality = {
    "rows_raw": int(len(df)),
    "missing_per_column": {k: int(v) for k, v in df.isnull().sum().items() if v > 0},
    "duplicates": int(df.duplicated().sum()),
    "invalid_time_rows": int(((df.Travel_Time_min <= 0) | (df.Travel_Time_min > 240)).sum()),
}
print(quality)

# ------------------------------------------------------------------ 3. CLEAN

d = df.drop_duplicates().copy()
d = d[(d.Travel_Time_min > 0) & (d.Travel_Time_min <= 240)]        # impossible commute times
d["Distance_km"] = d["Distance_km"].fillna(d.groupby(["Start_Area", "Destination"])["Distance_km"].transform("median"))
d["Distance_km"] = d["Distance_km"].fillna(d["Distance_km"].median())
for c in ["Traffic_Level", "Weather"]:                              # categorical -> mode
    d[c] = d[c].fillna(d[c].mode()[0])
d = d.reset_index(drop=True)
quality["rows_clean"] = int(len(d))

# ------------------------------------------------------------------ 4. FEATURE ENGINEERING
d["Peak_Hour"] = d["Reach_Hour"].between(8.5, 9.5).astype(int)
d["Weekend"] = (d["Day"] == "Sat").astype(int)
d["Traffic_Score"] = d["Traffic_Level"].map({"Low": 1, "Medium": 2, "High": 3})
d["Rain"] = (d["Weather"] == "Rain").astype(int)

num = ["Distance_km", "Reach_Hour", "Traffic_Score", "Peak_Hour", "Weekend", "Rain"]
cat = ["Transport_Mode", "Start_Area", "Destination"]
X = d[num + cat]; y = d["Travel_Time_min"]   # Cost_Rs excluded: it is an outcome of mode+distance, not an input
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)

pre = ColumnTransformer([("num", StandardScaler(), num),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])

# ------------------------------------------------------------------ 5. VISUALS (EDA)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3})
fig, ax = plt.subplots(figsize=(6, 4))
for m, g in d.groupby("Transport_Mode"):
    ax.scatter(g.Distance_km, g.Travel_Time_min, s=14, alpha=.6, label=m)
ax.set(xlabel="Distance (km)", ylabel="Travel time (min)", title="Distance vs Travel Time by Mode"); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_distance_vs_time.png"); plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
order = ["Low", "Medium", "High"]
ax.boxplot([d[d.Traffic_Level == t].Travel_Time_min for t in order], tick_labels=order)
ax.set(xlabel="Traffic level", ylabel="Travel time (min)", title="Traffic Level vs Travel Time")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_traffic_vs_time.png"); plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(d.Travel_Time_min, bins=30, color="#2a6f97"); ax.set(xlabel="Travel time (min)", ylabel="Count", title="Travel Time Distribution")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_time_distribution.png"); plt.close()

# ------------------------------------------------------------------ 6. MODELS
models = {
    "Linear Regression": Pipeline([("pre", pre), ("m", LinearRegression())]),
    "Random Forest": Pipeline([("pre", pre), ("m", RandomForestRegressor(n_estimators=200, random_state=SEED))]),
}
results, preds = {}, {}
for name, p in models.items():
    p.fit(Xtr, ytr); yp = p.predict(Xte); preds[name] = yp
    cv = cross_val_score(p, Xtr, ytr, cv=5, scoring="r2")
    mse = mean_squared_error(yte, yp)
    results[name] = dict(MAE=mean_absolute_error(yte, yp), MSE=mse, RMSE=float(np.sqrt(mse)),
                         R2=r2_score(yte, yp), CV_R2_mean=float(cv.mean()), CV_R2_std=float(cv.std()))
# hyperparameter tuning (optional advanced feature)
gs = GridSearchCV(Pipeline([("pre", pre), ("m", RandomForestRegressor(random_state=SEED))]),
                  {"m__n_estimators": [100, 300], "m__max_depth": [None, 8, 14], "m__min_samples_leaf": [1, 3]},
                  cv=5, scoring="neg_mean_absolute_error", n_jobs=-1).fit(Xtr, ytr)
best = gs.best_estimator_; yp = best.predict(Xte); preds["Random Forest (tuned)"] = yp
mse = mean_squared_error(yte, yp)
results["Random Forest (tuned)"] = dict(MAE=mean_absolute_error(yte, yp), MSE=mse, RMSE=float(np.sqrt(mse)),
                                        R2=r2_score(yte, yp), CV_R2_mean=float(-gs.best_score_), CV_R2_std=0.0)
results["Random Forest (tuned)"]["CV_R2_mean"] = float(cross_val_score(best, Xtr, ytr, cv=5, scoring="r2").mean())
best_params = gs.best_params_
print(pd.DataFrame(results).T.round(3))

# ------------------------------------------------------------------ 7. EVALUATION VISUALS
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
for ax, n in zip(axs, ["Linear Regression", "Random Forest (tuned)"]):
    ax.scatter(yte, preds[n], s=14, alpha=.6)
    lim = [0, max(yte.max(), preds[n].max()) + 5]; ax.plot(lim, lim, "r--")
    ax.set(xlabel="Actual (min)", ylabel="Predicted (min)", title=f"{n}\nR2={results[n]['R2']:.3f}")
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_actual_vs_predicted.png"); plt.close()

res = yte - preds["Random Forest (tuned)"]
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(preds["Random Forest (tuned)"], res, s=14, alpha=.6); ax.axhline(0, color="r", ls="--")
ax.set(xlabel="Predicted (min)", ylabel="Residual (min)", title="Residuals (tuned Random Forest)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_residuals.png"); plt.close()

# feature importance
names = best.named_steps["pre"].get_feature_names_out()
imp = pd.Series(best.named_steps["m"].feature_importances_, index=names).sort_values()
imp.index = [i.split("__")[1] for i in imp.index]
top = imp.tail(10)
fig, ax = plt.subplots(figsize=(6, 4)); top.plot.barh(ax=ax, color="#2a6f97")
ax.set(title="Top Feature Importances (Random Forest)", xlabel="Importance"); fig.tight_layout()
fig.savefig(f"{OUT}/fig6_feature_importance.png"); plt.close()
importances = {k: float(v) for k, v in imp.tail(6)[::-1].items()}

# ------------------------------------------------------------------ 8. RECOMMENDER + MATCHING
COST_KM = {"Bike": 2.5, "Car": 8.0, "Bus": 1.8, "Shared Auto": 4.0}   # Rs/km assumptions
def recommend(area, dest, dist, hour, traffic, weather, day="Mon", w_time=0.6):
    """Predict time for every mode, then rank by weighted (time, cost) score."""

    rows = []
    for m in COST_KM:
        r = pd.DataFrame([dict(Distance_km=dist, Reach_Hour=hour, Traffic_Score={"Low": 1, "Medium": 2, "High": 3}[traffic],
                               Peak_Hour=int(8.5 <= hour <= 9.5), Weekend=int(day == "Sat"), Rain=int(weather == "Rain"),
                               Transport_Mode=m, Start_Area=area, Destination=dest)])
        rows.append((m, float(best.predict(r)[0]), dist * COST_KM[m]))
    t = pd.DataFrame(rows, columns=["Mode", "Pred_Time_min", "Est_Cost_Rs"])
    nz = lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9)
    t["Score"] = w_time * nz(t.Pred_Time_min) + (1 - w_time) * nz(t.Est_Cost_Rs)
    return t.sort_values("Score").reset_index(drop=True)

def match(area, dest, hour, db="commute_data.txt"):
    """Original Java rule: same destination + area + reach time (reads the Java project's file if present)."""
    if not os.path.exists(db): return []
    out = []
    for line in open(db):
        p = line.strip().split(",")
        if len(p) >= 6 and p[2].lower() == dest.lower() and p[3].lower() == area.lower() and p[4].lower().startswith(f"{int(hour)}:"):
             out.append(p[0])
    return out

demo = recommend("Katraj", "PVG College", 6.5, 9.0, "High", "Rain", "Mon")
print(demo)
demo.to_csv(f"{OUT}/demo_recommendation.csv", index=False)

# ------------------------------------------------------------------ 9. SAVE
json.dump(dict(simulated=SIMULATED, quality=quality, results=results, best_params=best_params,
               importances=importances, demo=demo.round(1).to_dict("records"),
               n_train=len(Xtr), n_test=len(Xte),
               time_stats=d.Travel_Time_min.describe().round(1).to_dict(),
               traffic_means=d.groupby("Traffic_Level").Travel_Time_min.mean().round(1).to_dict(),
               mode_means=d.groupby("Transport_Mode").Travel_Time_min.mean().round(1).to_dict(),
               corr_dist_time=float(d.Distance_km.corr(d.Travel_Time_min))),
          open(f"{OUT}/results.json", "w"), indent=2)
