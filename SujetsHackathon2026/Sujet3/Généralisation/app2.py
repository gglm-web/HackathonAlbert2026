import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
import random
from math import radians, cos, sin, asin, sqrt

# ==================== CONFIGURATION ====================
st.set_page_config(page_title="Surveillance Maritime Avancée", layout="wide")
st.title("🔍 Système d'Identification et de Surveillance Maritime")

# --- FONCTION HAVERSINE POUR LES ÉCARTS GPS ---
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * asin(sqrt(a)) * 6371

# ==================== CHARGEMENT DES DONNÉES ====================
@st.cache_data
def load_data():
    try:
        df_c = pd.read_csv("caracterisation.csv")
        df_a = pd.read_csv("ais_data_large.csv")
        df_a['timestamp'] = pd.to_datetime(df_a['timestamp'])
        return df_c, df_a
    except Exception as e:
        st.error(f"Erreur de chargement des fichiers CSV : {e}")
        st.stop()

df_ref, df_ais_all = load_data()

# Initialisation du session_state pour la génération automatique
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
st.sidebar.header("⚙️ Contrôles")
st.sidebar.button("🎲 Générer une signature automatique", on_click=generate_random_signature, use_container_width=True)
st.sidebar.divider()
st.sidebar.info(f"Base de données : {len(df_ref)} navires référencés.")

# ==================== FORMULAIRE DE SAISIE ====================
st.header("📡 Capture du Signal Radio")
col_in1, col_in2 = st.columns(2)
with col_in1:
    freq = st.number_input("Fréquence (MHz)", value=st.session_state.input_vals['freq'], format="%.4f")
    bw = st.number_input("Bande passante (kHz)", value=st.session_state.input_vals['bw'], format="%.2f")
    power = st.number_input("Puissance (Watts)", value=st.session_state.input_vals['power'], format="%.2f")
with col_in2:
    sig_strength = st.number_input("Force du signal (dBm)", value=st.session_state.input_vals['sig'], format="%.2f")
    snr = st.number_input("SNR (dB)", value=st.session_state.input_vals['snr'], format="%.2f")
    noise = st.number_input("Bruit (dBm)", value=st.session_state.input_vals['noise'], format="%.2f")

# ==================== ANALYSE ET RÉSULTATS ====================
if st.button("🔍 Lancer l'Analyse Complète", type="primary", use_container_width=True):
    
    # 1. MATCHING RADIO
    features = ['freq_mean_mhz', 'bandwidth_mean_khz', 'power_mean_watts', 'signal_strength_mean_dbm', 'snr_mean_db', 'noise_level_mean_dbm']
    new_sig_data = [freq, bw, power, sig_strength, snr, noise]
    
    scaler = StandardScaler()
    ref_scaled = scaler.fit_transform(df_ref[features])
    new_scaled = scaler.transform(pd.DataFrame([new_sig_data], columns=features))
    distances = np.linalg.norm(ref_scaled - new_scaled, axis=1)
    
    df_result = df_ref.copy()
    df_result['similarity_%'] = (np.exp(-distances/2) * 100).round(1)
    best = df_result.sort_values('similarity_%', ascending=False).iloc[0]

    st.divider()

    # --- SECTION A : IDENTITÉ ET PAVILLONS ---
    st.subheader(f"🆔 Identification : MMSI {int(best['mmsi'])}")
    
    id_c1, id_c2, id_c3, id_c4 = st.columns(4)
    with id_c1:
        st.metric("Type", best['type'])
        if best['is_suspicious']: st.error("🚨 FICHÉ SUSPECT")
        else: st.success("✅ FICHÉ NORMAL")
    with id_c2:
        st.metric("Tonnage", f"{best['gross_tonnage']:,} GT")
    with id_c3:
        st.metric("Pavillon Supposé", best['pavillon_suppose'])
    with id_c4:
        st.metric("Pavillon Déclaré", best['flag'])
        if best['pavillon_suppose'] != best['flag']:
            st.warning("⚠️ INCOHÉRENCE PAVILLON")
    
    # --- SECTION B : DÉTAILS DE LA SIGNATURE RADIO ---
    st.divider()
    st.subheader("📊 Analyse de la Signature et Fiabilité")
    
    # Métriques de fiabilité
    conf_c1, conf_c2, conf_c3 = st.columns([1, 1, 2])
    conf_c1.metric("Score de Fiabilité", f"{best['similarity_%']}%")
    
    # Calcul des écarts détaillés
    tolerances = {'freq_mean_mhz': 0.05, 'bandwidth_mean_khz': 3.0, 'power_mean_watts': 30.0, 'signal_strength_mean_dbm': 6.0, 'snr_mean_db': 5.0, 'noise_level_mean_dbm': 8.0}
    comp_df = pd.DataFrame({
        'Paramètre': features,
        'Valeur Captée': [round(x, 3) for x in new_sig_data],
        'Valeur Référence': [round(best[f], 3) for f in features],
        'Écart Absolu': [round(abs(new_sig_data[i] - best[features[i]]), 3) for i in range(len(features))],
        'Tolérance Max': [tolerances[f] for f in features]
    })
    comp_df['Conformité'] = comp_df['Écart Absolu'] <= comp_df['Tolérance Max']

    # Affichage du tableau stylisé (Utilisation de .map pour éviter AttributeError)
    def style_conformity(row):
        color = 'background-color: rgba(46, 204, 113, 0.15)' if row['Conformité'] else 'background-color: rgba(231, 76, 60, 0.25)'
        return [color] * len(row)

    st.dataframe(
        comp_df.style.apply(style_conformity, axis=1)
        .map(lambda x: 'color: #e74c3c; font-weight: bold;' if x == False else 'color: #2ecc71;', subset=['Conformité'])
        .format({'Conformité': lambda x: "DANS LES SEUILS" if x else "HORS TOLÉRANCE"}),
        use_container_width=True, hide_index=True
    )

    # Graphique Radar de comparaison
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=new_sig_data, theta=features, fill='toself', name='Signal Capté'))
    fig_radar.add_trace(go.Scatterpolar(r=best[features].values, theta=features, fill='toself', name=f'Référence MMSI {int(best["mmsi"])}'))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Comparaison visuelle des signatures")
    st.plotly_chart(fig_radar, use_container_width=True)

    # --- SECTION C : COMPORTEMENT ET HISTORIQUE AIS ---
    st.divider()
    st.subheader("📍 Analyse de Trajectoire et Statut AIS")
    
    vessel_history = df_ais_all[df_ais_all['mmsi'] == int(best['mmsi'])].sort_values('timestamp')
    
    if not vessel_history.empty:
        # Statut AIS Active
        last_status = vessel_history.iloc[-1]['ais_active']
        if last_status:
            st.success("🛰️ AIS ACTIF : Le navire émet sa position en temps réel.")
        else:
            st.error("🌑 AIS INACTIF : Dark Activity détectée (transpondeur coupé).")

        # Calcul vitesse entre points
        vessel_history['speed_kmh'] = 0.0
        traj_error = False
        for i in range(1, len(vessel_history)):
            dist = haversine(vessel_history.iloc[i-1]['longitude'], vessel_history.iloc[i-1]['latitude'],
                             vessel_history.iloc[i]['longitude'], vessel_history.iloc[i]['latitude'])
            time_h = (vessel_history.iloc[i]['timestamp'] - vessel_history.iloc[i-1]['timestamp']).total_seconds() / 3600
            speed = dist / time_h if time_h > 0 else 0
            vessel_history.iloc[i, vessel_history.columns.get_loc('speed_kmh')] = speed
            if speed > 60: traj_error = True # Seuil de 60 km/h pour navire marchand

        if traj_error:
            st.error("🚨 ALERTE TRAJECTOIRE : Vitesses incohérentes détectées entre les positions.")

        # Carte
        fig_map = px.scatter_mapbox(vessel_history, lat="latitude", lon="longitude", color="speed_kmh",
                                   hover_name="timestamp", zoom=2, height=500,
                                   title=f"Historique des positions - MMSI {int(best['mmsi'])}")
        fig_map.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("Aucune donnée AIS disponible pour ce navire dans la base historique.")