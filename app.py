import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import linregress
import h2o
import datetime

# 1) Démarrage H2O
h2o.init(max_mem_size="2G")

# 2) Chargement des MOJOs
clf_mojo = h2o.upload_mojo("XGBoost_classification_model.zip")
reg_mojo = h2o.upload_mojo("XGBoost_regression_model.zip")

st.title("⚙️ Alerte panne imminente")

# 3) Upload CSV bruts
f = st.file_uploader("Charger CSV mesures brutes", type="csv")
if not f:
    st.stop()
df_raw = pd.read_csv(f)

# paramètres de segmentation codés en dur
WINDOW    = 100    # taille de la fenêtre (points)
STEP      = 50     # pas de la fenêtre (points)
DT        = 0.01   # pas de temps (s)
THRESHOLD = 3.0    # seuil pour Cest (ppm)

# 4) Feature‐engineering
def compute_segment_features(seg):
    feats = {}
    n = len(seg)
    t = np.arange(n) * DT
    for sig in ["I0","A0","K","Tcell","Pline","A2f","Cest"]:
        x = seg[sig].values
        mu, std = x.mean(), x.std(ddof=0)
        mx, mn = x.max(), x.min()
        slope, *_ = linregress(t, x)
        feats[f"mean_{sig}"]  = mu
        feats[f"std_{sig}"]   = std
        feats[f"slope_{sig}"] = slope
        if sig in ("Cest","I0","Pline"):
            feats[f"max_{sig}"]  = mx
            feats[f"drop_{sig}"] = mx - mn
    feats["timeOver3ppm"] = np.sum(seg["Cest"] > THRESHOLD) * DT
    return feats

# 5) Calcul de toutes les fenêtres
rows = []
for start in range(0, len(df_raw) - WINDOW + 1, STEP):
    seg = df_raw.iloc[start : start + WINDOW]
    rows.append(compute_segment_features(seg))
df_feats = pd.DataFrame(rows)

# 6) Prédictions et affichage des alertes uniquement
# …
# 6) Prédictions et affichage des alertes uniquement
if st.button("📡 Vérifier panne", key="alert"):
    # 6.1) Passez les features en H2OFrame et prédisez
    hf       = h2o.H2OFrame(df_feats)
    pred_clf = clf_mojo.predict(hf).as_data_frame()["predict"]
    pred_reg = reg_mojo.predict(hf).as_data_frame()["predict"]

    # 6.2) Construisez la liste des alertes valides
    alerts = []
    for idx, (cls, ttf) in enumerate(zip(pred_clf, pred_reg)):
        # filtrez : pas de NaN, horizon ≥ 0, classe non "None"
        if pd.isna(ttf) or ttf < 0 or cls in (None, "None"):
            continue
        # convertissez seconds → jours entiers
        days = int(ttf // 86400)
        if days > 6:
            continue  # on ignore tout au-delà de 6 jours

        # formatez l’horizon
        if days == 0:
            td = datetime.timedelta(seconds=ttf)
            h  = td.seconds // 3600
            m  = (td.seconds % 3600) // 60
            lbl = f"dans {h} h {m} min"
        else:
            lbl = f"dans {days} jour{'s' if days > 1 else ''}"

        alerts.append((idx, cls, lbl))

    # 6.3) Affichez un message unique selon qu’il y ait ou non des alertes
    if not alerts:
        st.success("✅ Aucune panne dans les 6 jours à venir.")
    else:
        st.subheader("⚠️ Alertes de pannes prévues :")
        for idx, cls, lbl in alerts:
            st.error(f"• Segment #{idx} → Panne « {cls} » {lbl}")

