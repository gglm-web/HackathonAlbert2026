import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import random
from math import radians, cos, sin, asin, sqrt
import plotly.express as px
# ==================== CONFIGURATION ====================
st.set_page_config(page_title="Signature Radio Navire", layout="wide")
st.title("🔍 Analyse de Signature Radio Maritime")

# ==================== CHARGEMENT DES DONNÉES ====================

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("caracterisation.csv")
        return df
    except FileNotFoundError:
        st.error("❌ Fichier 'caracterisation.csv' introuvable.")
        st.stop()

df_ref = load_data()

# Vérification des colonnes (incluant les nouvelles)
required_cols = [
    'mmsi', 'freq_mean_mhz', 'bandwidth_mean_khz', 'power_mean_watts',
    'signal_strength_mean_dbm', 'snr_mean_db', 'noise_level_mean_dbm', 
    'is_suspicious', 'gross_tonnage', 'type'
]

missing = [col for col in required_cols if col not in df_ref.columns]
if missing:
    st.error(f"Colonnes manquantes dans le CSV : {missing}")
    st.stop()

# Initialisation session_state pour les entrées
if 'input_vals' not in st.session_state:
    st.session_state.input_vals = {'freq': 158.0, 'bw': 25.0, 'power': 200.0, 'sig': -80.0, 'snr': 30.0, 'noise': -90.0}

def generate_random_signature():
    vessel = df_ref.sample(n=1).iloc[0]
    st.session_state.input_vals = {
        'freq': vessel['freq_mean_mhz'] + random.uniform(-0.02, 0.02),
        'bw': vessel['bandwidth_mean_khz'] + random.uniform(-1.0, 1.0),
        'power': max(0, vessel['power_mean_watts'] + random.uniform(-10.0, 10.0)),
        'sig': vessel['signal_strength_mean_dbm'] + random.uniform(-2.0, 2.0),
        'snr': vessel['snr_mean_db'] + random.uniform(-1.5, 1.5),
        'noise': vessel['noise_level_mean_dbm'] + random.uniform(-2.0, 2.0)
    }

# ==================== INTERFACE LATÉRALE ====================
st.sidebar.header("⚙️ Actions")
st.sidebar.button("🎲 Générer une signature automatique", on_click=generate_random_signature, use_container_width=True)
st.sidebar.divider()
st.sidebar.subheader("Base de données")
st.sidebar.dataframe(df_ref[['mmsi', 'type', 'gross_tonnage']].head(10), hide_index=True)

# ==================== FORMULAIRE DE SAISIE ====================
st.header("📡 Signal Radio Capté")
c1, c2 = st.columns(2)
with c1:
    freq = st.number_input("Fréquence (MHz)", value=st.session_state.input_vals['freq'], format="%.4f")
    bw = st.number_input("Bande passante (kHz)", value=st.session_state.input_vals['bw'])
    power = st.number_input("Puissance (Watts)", value=st.session_state.input_vals['power'])
with c2:
    sig_strength = st.number_input("Force du signal (dBm)", value=st.session_state.input_vals['sig'])
    snr = st.number_input("SNR (dB)", value=st.session_state.input_vals['snr'])
    noise = st.number_input("Bruit (dBm)", value=st.session_state.input_vals['noise'])

# ==================== ANALYSE DE CORRESPONDANCE ====================
if st.button("🔍 Lancer l'identification", type="primary", use_container_width=True):
    # Paramètres utilisés pour le calcul de similarité (radio uniquement)
    features = ['freq_mean_mhz', 'bandwidth_mean_khz', 'power_mean_watts', 'signal_strength_mean_dbm', 'snr_mean_db', 'noise_level_mean_dbm']
    new_sig_data = [freq, bw, power, sig_strength, snr, noise]
    
    # Calcul de distance
    scaler = StandardScaler()
    ref_scaled = scaler.fit_transform(df_ref[features])
    new_scaled = scaler.transform(pd.DataFrame([new_sig_data], columns=features))
    distances = np.linalg.norm(ref_scaled - new_scaled, axis=1)
    
    # Résultat
    df_result = df_ref.copy()
    df_result['similarity_%'] = (np.exp(-distances/2) * 100).round(1)
    best = df_result.sort_values('similarity_%', ascending=False).iloc[0]

    st.divider()

    # --- 1. FICHE D'IDENTITÉ DU NAVIRE ---
    st.subheader("🆔 Fiche d'identité du navire identifié")
    
    idx_col1, idx_col2, idx_col3, idx_col4 = st.columns(4)
    
    with idx_col1:
        st.metric("MMSI", int(best['mmsi']))
    with idx_col2:
        st.metric("Type de Bateau", best['type'])
    with idx_col3:
        st.metric("Tonnage (GT)", f"{best['gross_tonnage']:,}")
    with idx_col4:
        # Affichage du statut suspect
        if best['is_suspicious']:
            st.error("⚠️ STATUT : SUSPECT")
        else:
            st.success("✅ STATUT : NORMAL")

    # --- 2. ANALYSE DES ÉCARTS ---
    st.subheader("📊 Analyse des écarts radio")
    
    tolerances = {'freq_mean_mhz': 0.05, 'bandwidth_mean_khz': 3.0, 'power_mean_watts': 30.0, 'signal_strength_mean_dbm': 6.0, 'snr_mean_db': 5.0, 'noise_level_mean_dbm': 8.0}
    
    comp_df = pd.DataFrame({
        'Paramètre': features,
        'Capté': [round(x, 3) for x in new_sig_data],
        'Référence': [round(best[f], 3) for f in features],
        'Écart': [round(abs(new_sig_data[i] - best[features[i]]), 3) for i in range(len(features))],
        'Tolérance': [tolerances[f] for f in features]
    })
    comp_df['Conformité'] = comp_df['Écart'] <= comp_df['Tolérance']

    # Liste textuelle des anomalies pour mise en évidence
    anomalies = comp_df[comp_df['Conformité'] == False]
    if not anomalies.empty:
        st.warning(f"Identification à {best['similarity_%']}% mais avec des anomalies sur {len(anomalies)} paramètre(s).")
    
    # Affichage du tableau avec .map()
    def style_rows(row):
        color = 'background-color: rgba(46, 204, 113, 0.1)' if row['Conformité'] else 'background-color: rgba(231, 76, 60, 0.2)'
        return [color] * len(row)

    st.dataframe(
        comp_df.style.apply(style_rows, axis=1)
        .map(lambda x: 'font-weight: bold; color: #e74c3c;' if x == False else '', subset=['Conformité'])
        .format({'Conformité': lambda x: "✅ OK" if x else "❌ HORS SEUIL"}),
        use_container_width=True, hide_index=True
    )

    # --- 3. VISUALISATION ---
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=new_sig_data, theta=features, fill='toself', name='Signal Capté'))
    fig.add_trace(go.Scatterpolar(r=best[features].values, theta=features, fill='toself', name=f'Réf MMSI {int(best["mmsi"])}'))
    fig.update_layout(title="Comparaison des Signatures (Radar)")
    st.plotly_chart(fig, use_container_width=True)

    # Message final si suspect
    if best['is_suspicious']:
        st.sidebar.error(f"🚨 ALERTE : Le navire {int(best['mmsi'])} est sous surveillance !")
from math import radians, cos, sin, asin, sqrt
import plotly.express as px

# --- FONCTION DE CALCUL DE DISTANCE (HAVERSINE) ---
def haversine(lon1, lat1, lon2, lat2):
    # Convertit les degrés décimaux en radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # Formule Haversine
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Rayon de la terre en km
    return c * r

# ==================== CHARGEMENT DES DONNÉES AIS ====================
@st.cache_data
def load_ais_history():
    try:
        df_ais = pd.read_csv("ais_data_large.csv")
        df_ais['timestamp'] = pd.to_datetime(df_ais['timestamp'])
        return df_ais
    except Exception as e:
        st.error(f"Erreur chargement ais_data_large.csv: {e}")
        return None

df_ais_all = load_ais_history()

# ==================== ANALYSE DE TRAJECTOIRE (À mettre après l'identification) ====================
if 'best' in locals(): # On vérifie qu'un bateau a été identifié
    st.divider()
    st.header("📍 Suivi des mouvements (Données AIS)")
    
    # 1. Filtrage sur le MMSI identifié
    vessel_history = df_ais_all[df_ais_all['mmsi'] == int(best['mmsi'])].sort_values('timestamp')
    
    if vessel_history.empty:
        st.info("Aucun historique de position trouvé pour ce MMSI dans ais_data_large.csv")
    else:
        # 2. Détection d'incohérences (Vitesse aberrante)
        vessel_history['dist_km'] = 0.0
        vessel_history['time_diff_h'] = 0.0
        vessel_history['speed_kmh'] = 0.0
        inconsistency_detected = False
        
        for i in range(1, len(vessel_history)):
            p1 = vessel_history.iloc[i-1]
            p2 = vessel_history.iloc[i]
            
            # Calcul distance et temps
            dist = haversine(p1['longitude'], p1['latitude'], p2['longitude'], p2['latitude'])
            tdiff = (p2['timestamp'] - p1['timestamp']).total_seconds() / 3600
            
            if tdiff > 0:
                speed = dist / tdiff
                vessel_history.iloc[i, vessel_history.columns.get_loc('speed_kmh')] = speed
                # Si vitesse > 60 km/h (~32 nœuds) pour un cargo/tanker, c'est louche
                if speed > 60: 
                    inconsistency_detected = True

        # 3. Affichage des Alertes de Trajectoire
        if inconsistency_detected:
            st.error("🚨 ALERTE TRAJECTOIRE : Positions incohérentes détectées !")
            st.write("Le navire semble s'être déplacé à une vitesse impossible entre deux relevés. Suspicion de spoofing GPS ou erreur de capteur.")
        else:
            st.success("✅ Trajectoire cohérente avec les capacités du navire.")

        # 4. Carte interactive
        fig_map = px.scatter_mapbox(
            vessel_history, 
            lat="latitude", 
            lon="longitude", 
            hover_name="timestamp",
            hover_data=["speed_kmh", "status"],
            color="speed_kmh",
            color_continuous_scale=px.colors.cyclical.IceFire,
            zoom=2, 
            height=500
        )
        fig_map.update_layout(mapbox_style="carto-positron")
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        
        # Ajout de la ligne de trajectoire
        fig_map.add_trace(go.Scattermapbox(
            mode = "lines",
            lon = vessel_history['longitude'],
            lat = vessel_history['latitude'],
            marker = {'size': 10},
            name = "Trajet"
        ))

        st.plotly_chart(fig_map, use_container_width=True)

        # 5. Tableau des dernières positions
        with st.expander("Voir le détail des positions"):
            st.dataframe(vessel_history[['timestamp', 'latitude', 'longitude', 'status', 'speed_kmh']].tail(10))