import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from joblib import load
except Exception:
    load = None

# -------------------- PAGE --------------------
st.set_page_config(page_title="Pokémon Battle Predictor", page_icon="⚡", layout="wide")

STAT_COLS = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]

# -------------------- DARK POKÉMON CSS --------------------
st.markdown(
    """
<style>
.stApp{
  background: radial-gradient(circle at 15% 10%, rgba(255,214,0,0.12) 0%, rgba(0,0,0,0) 35%),
              radial-gradient(circle at 85% 15%, rgba(0,163,255,0.12) 0%, rgba(0,0,0,0) 35%),
              linear-gradient(180deg, #0b0f1a 0%, #070a12 100%);
  color: #e9eefc;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.hero{ text-align:center; padding: 18px 10px 8px 10px; }
.hero-title{
  font-size: 44px; font-weight: 900; letter-spacing: 1px;
  color: #f3f6ff; text-shadow: 0 0 18px rgba(255,214,0,.25);
}
.hero-sub{ font-size: 15px; opacity: .85; margin-top: 6px; }

.pill{
  display:inline-block; padding: 6px 10px; border-radius: 999px;
  background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.08);
  margin: 4px 6px 0 0; font-size: 13px; backdrop-filter: blur(8px);
}

.card{
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 16px 16px 12px 16px;
  box-shadow: 0 18px 40px rgba(0,0,0,.35);
  backdrop-filter: blur(10px);
}
.card-title{ font-weight: 900; font-size: 18px; margin-bottom: 8px; }
.red{color:#ff4d6d;}
.blue{color:#4dabff;}

.vs{
  text-align:center; font-size: 46px; font-weight: 900; opacity: .85;
  margin-top: 34px; color: #f3f6ff; text-shadow: 0 0 18px rgba(0,163,255,.18);
}

.winbox{
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 16px;
  box-shadow: 0 18px 40px rgba(0,0,0,.35);
  backdrop-filter: blur(10px);
}

.stButton>button{
  border-radius: 14px !important;
  padding: 10px 16px !important;
  font-weight: 900 !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  background: linear-gradient(90deg, rgba(255,214,0,0.22), rgba(0,163,255,0.18)) !important;
  color: #f3f6ff !important;
}
.stButton>button:hover{
  transform: translateY(-1px);
  box-shadow: 0 12px 28px rgba(0,0,0,.35);
}

section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.03) 100%);
  border-right: 1px solid rgba(255,255,255,0.08);
}
.small-note{opacity:.75;font-size:12px;}
hr{border-color: rgba(255,255,255,0.10);}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------- HELPERS --------------------
@st.cache_data
def load_csv_from_path(p: str) -> pd.DataFrame:
    return pd.read_csv(p)

def safe_joblib_load(path: str):
    if load is None:
        return None
    p = Path(path)
    return load(p) if p.exists() else None

def get_row(df: pd.DataFrame, name: str):
    r = df[df["Name"].astype(str) == str(name)]
    return None if r.empty else r.iloc[0]

def build_features(first_row, second_row):
    # 12 features: First stats + Second stats (MUST match training order)
    x = [first_row[c] for c in STAT_COLS] + [second_row[c] for c in STAT_COLS]
    return np.array(x, dtype=float).reshape(1, -1)

def pill(txt): 
    return f"<span class='pill'>{txt}</span>"

def render_card(label, row, color_class=""):
    if row is None:
        st.error("Pokémon not found in CSV.")
        return

    t1 = row["Type 1"] if "Type 1" in row.index else ""
    t2 = row["Type 2"] if "Type 2" in row.index else ""
    pid = row["#"] if "#" in row.index else ""

    pills = ""
    if str(pid) not in ["nan", ""]:
        pills += pill(f"ID: <b>{pid}</b>")
    if str(t1) not in ["nan", ""]:
        pills += pill(f"Type1: <b>{t1}</b>")
    if str(t2) not in ["nan", ""]:
        pills += pill(f"Type2: <b>{t2}</b>")
    pills += "<br/>"
    pills += pill(f"HP: <b>{row['HP']}</b>")
    pills += pill(f"Atk: <b>{row['Attack']}</b>")
    pills += pill(f"Def: <b>{row['Defense']}</b>")
    pills += pill(f"SpA: <b>{row['Sp. Atk']}</b>")
    pills += pill(f"SpD: <b>{row['Sp. Def']}</b>")
    pills += pill(f"Speed: <b>{row['Speed']}</b>")

    st.markdown(
        f"""
    <div class="card">
      <div class="card-title {color_class}">{label} — {row["Name"]}</div>
      <div>{pills}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# -------------------- HEADER --------------------
st.markdown(
    """
<div class="hero">
  <div class="hero-title">⚡ Pokémon Battle Predictor ⚡</div>
  <div class="hero-sub">Choose two Pokémon — model predicts the winner 🏆</div>
</div>
""",
    unsafe_allow_html=True,
)

# -------------------- SIDEBAR --------------------
st.sidebar.title("⚙️ Setup")
st.sidebar.markdown("Upload files (recommended) to avoid path issues.")

uploaded_pokemon_csv = st.sidebar.file_uploader("Upload pokemon.csv", type=["csv"], key="pokemon_csv_main")

model_path = st.sidebar.text_input("model.joblib path (optional)", value="model.joblib")
scaler_path = st.sidebar.text_input("scaler.joblib path (optional)", value="scaler.joblib")

st.sidebar.markdown("---")
label_hint = st.sidebar.selectbox(
    "Winner meaning in your model",
    ["1 = First wins, 0 = Second wins", "1 = Second wins, 0 = First wins"],
    index=0,
)

use_scaler = st.sidebar.checkbox("Use scaler.joblib (only if you used it in training)", value=False)
debug_mode = st.sidebar.checkbox("Debug mode (show pred/probabilities)", value=False)

st.sidebar.markdown(
    "<div class='small-note'>⚠️ IMPORTANT: The 12 input features MUST match the training order:<br/>First stats then Second stats.</div>",
    unsafe_allow_html=True,
)

# -------------------- LOAD CSV --------------------
poke_df = None

if uploaded_pokemon_csv is not None:
    try:
        uploaded_pokemon_csv.seek(0)
        poke_df = pd.read_csv(uploaded_pokemon_csv)
    except pd.errors.EmptyDataError:
        st.error("Uploaded pokemon.csv is empty.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to read uploaded pokemon.csv: {e}")
        st.stop()
else:
    # auto-detect local
    if Path("pokemon.csv").exists():
        poke_df = load_csv_from_path("pokemon.csv")
    elif Path("/mnt/data/pokemon.csv").exists():  # colab
        poke_df = load_csv_from_path("/mnt/data/pokemon.csv")

if poke_df is None:
    st.error("Could not find pokemon.csv. Upload it in sidebar OR place it next to app.py as 'pokemon.csv'.")
    st.stop()

# Validate columns
need = ["Name"] + STAT_COLS
missing = [c for c in need if c not in poke_df.columns]
if missing:
    st.error(f"pokemon.csv missing columns: {missing}")
    st.stop()

names = poke_df["Name"].dropna().astype(str).tolist()
if len(names) < 2:
    st.error("Not enough Pokémon in CSV.")
    st.stop()

# -------------------- LOAD MODEL/SCALER --------------------
model = safe_joblib_load(model_path)
scaler = safe_joblib_load(scaler_path) if use_scaler else None

if model is None:
    st.warning("Model not found. UI works, but prediction needs model.joblib (and optional scaler.joblib).")

# -------------------- PICKERS --------------------
left, mid, right = st.columns([4.8, 1.0, 4.8], vertical_alignment="top")
with left:
    first_name = st.selectbox("🟥 First Pokémon", names, index=0)
with mid:
    st.markdown("<div class='vs'>VS</div>", unsafe_allow_html=True)
with right:
    second_name = st.selectbox("🟦 Second Pokémon", names, index=1 if len(names) > 1 else 0)

first_row = get_row(poke_df, first_name)
second_row = get_row(poke_df, second_name)

st.write("")
c1, c2 = st.columns(2)
with c1:
    render_card("🟥 First", first_row, "red")
with c2:
    render_card("🟦 Second", second_row, "blue")

st.write("")

# -------------------- PREDICTION --------------------
st.markdown("<div class='winbox'>", unsafe_allow_html=True)
p1, p2, p3 = st.columns([2.4, 1.2, 2.4], vertical_alignment="center")
with p2:
    btn = st.button("🔮 Predict Winner", use_container_width=True)

if btn:
    if first_name == second_name:
        st.warning("You selected the same Pokémon for both sides. Pick two different Pokémon.")
    elif model is None:
        st.error("Add model.joblib to your project folder (or set correct path in sidebar).")
    else:
        X = build_features(first_row, second_row)

        # apply scaler if used during training
        if scaler is not None:
            try:
                X = scaler.transform(X)
            except Exception as e:
                st.warning(f"Scaler transform failed: {e}")

        try:
            pred = int(model.predict(X)[0])

            # ✅ DEBUG LINE (this is the exact place)
            if debug_mode:
                st.write("DEBUG pred:", pred)

            # Mapping based on your sidebar choice
            if "1 = First wins" in label_hint:
                first_wins = (pred == 1)
            else:
                first_wins = (pred == 0)

            if first_wins:
                st.success(f"🏆 Predicted Winner: **{first_name}** (First Pokémon)")
            else:
                st.success(f"🏆 Predicted Winner: **{second_name}** (Second Pokémon)")

            # Optional probabilities
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]  # [P(class0), P(class1)]
                if debug_mode:
                    st.write("DEBUG proba [class0,class1]:", proba)

                # Show "First wins" probability depending on label meaning
                if len(proba) >= 2:
                    if "1 = First wins" in label_hint:
                        p_first = float(proba[1])
                    else:
                        # if label meaning is swapped, "first wins" is class 0
                        p_first = float(proba[0])

                    st.info(f"📊 Win Probability (First Pokémon): **{p_first*100:.2f}%**")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div class='small-note'>Dark Pokémon theme UI • Streamlit</div>", unsafe_allow_html=True)