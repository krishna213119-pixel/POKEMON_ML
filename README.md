⚡ Pokémon Battle Predictor – ML Web App
<p align="center"> <img src="https://img.shields.io/badge/Streamlit-Deployed-red?style=for-the-badge&logo=streamlit"> <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python"> <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge&logo=scikitlearn"> <img src="https://img.shields.io/badge/Status-Live-success?style=for-the-badge"> </p> <p align="center"> A Machine Learning powered web application that predicts the winner between two Pokémon using battle statistics. </p>
🚀 Live Application

👉 Try it here:
https://pokemonml-3i3femewbgxx4sq6wphmvk.streamlit.app/

📌 Features

🤖 Trained Machine Learning model (Binary Classification)

⚔️ Predicts winner between two Pokémon

📊 Displays win probability

🎨 Custom Dark Pokémon-themed UI

⚙️ Optional Scaler integration

🐞 Debug mode for prediction insights

☁️ Deployed on Streamlit Cloud

🛠 Tech Stack
Category	Tools Used
Language	Python 3.11
ML	Scikit-learn
Data	Pandas, NumPy
Model Storage	Joblib
UI	Streamlit
Deployment	Streamlit Cloud
Version Control	Git & GitHub
🧠 How It Works
1️⃣ Feature Construction

Each prediction uses 12 input features:

First Pokémon:
[HP, Attack, Defense, Sp. Atk, Sp. Def, Speed]

Second Pokémon:
[HP, Attack, Defense, Sp. Atk, Sp. Def, Speed]

These are combined and passed to the trained model.

2️⃣ Model Prediction

The ML model outputs:

1 → First Pokémon wins

0 → Second Pokémon wins

If available, prediction probability is also displayed.

📂 Project Structure
POKEMON_ML/
│
├── app.py              # Main Streamlit application
├── pokemon.csv         # Pokémon stats dataset
├── combats.csv         # Battle history dataset
├── model.joblib        # Trained ML model
├── scaler.joblib       # Optional scaler
├── requirements.txt    # Dependencies
├── runtime.txt         # Python version for deployment
└── README.md
