import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
import random
from math import radians, cos, sin, asin, sqrt

# ==================== CONFIGURATION ====================
st.set_page_config(page_title="Centre de Surveillance Maritime", layout="wide", initial_sidebar_state="expanded")

# ==================== DESIGN MILITAIRE PREMIUM ====================
st.markdown("""
<style>
/* ... tout ton CSS reste identique ... */
</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown("""
<div class="main-header">
    <h1> MARINE NATIONALE</h1>
    <p>CENTRE DE SURVEILLANCE MARITIME • ANALYSE RADIO & AIS • ALBERT SCHOOL</p>
</div>
""", unsafe_allow_html=True)

# ==================== LOGO LOCAL ====================
st.sidebar.image("logo_marine.svg", width=160)

# ==================== RÉFÉRENTIELS ====================
VITESSE_MAX_NAVIRE = {
    'Container Ship': 25.0, 'Passenger Ship': 30.0, 'Fishing Vessel': 15.0,
    'Tanker': 16.0, 'Bulk Carrier': 15.0, 'General Cargo': 18.0
}

DISTANCE_ANORMALE_NM = 150.0

def haversine_knots(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * asin(sqrt(a)) * 3440.06

# ==================== CHARGEMENT DES DONNÉES ====================
@st.cache_data
def load_data():
    try:
        df_c = pd.read_csv("caracterisation.csv")
        df_a = pd.read_csv("ais_data_large.csv")
        df_a['timestamp'] = pd.to_datetime(df_a['timestamp'])
        return df_c, df_a
    except Exception as e:
        st.error(f"Erreur de chargement des fichiers : {e}")
        st.stop()

df_ref, df_ais_all = load_data()

# ==================== SESSION STATE & SIMULATION ====================
if 'input_vals' not in st.session_state:
    st.session_state.input_vals = {'freq': 158.0, 'bw': 25.0, 'power': 200.0, 
                                   'sig': -80.0, 'snr': 30.0, 'noise': -90.0}

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

st.sidebar.header(" Simulation de signal")
st.sidebar.button(" Générer un signal aléatoire", on_click=generate_random_signature, use_container_width=True)
st.sidebar.divider()
st.sidebar.markdown("**Classification** : CONFIDENTIEL / MARINE NATIONALE / DONNEES GENEREES AUTOMATIQUEMENT")

# ==================== SAISIE ====================
st.header(" 📡 CAPTURE DU SIGNAL RADIO")
col_in1, col_in2 = st.columns(2)
with col_in1:
    freq = st.number_input("Fréquence (MHz)", value=st.session_state.input_vals['freq'], format="%.4f")
    bw = st.number_input("Bande passante (kHz)", value=st.session_state.input_vals['bw'])
    power = st.number_input("Puissance (Watts)", value=st.session_state.input_vals['power'])
with col_in2:
    sig_strength = st.number_input("Force du signal (dBm)", value=st.session_state.input_vals['sig'])
    snr = st.number_input("SNR (dB)", value=st.session_state.input_vals['snr'])
    noise = st.number_input("Bruit (dBm)", value=st.session_state.input_vals['noise'])

# ==================== ANALYSE ====================
if st.button("🔴 LANCER L'ANALYSE OPÉRATIONNELLE", type="primary", use_container_width=True):
    
    features = ['freq_mean_mhz', 'bandwidth_mean_khz', 'power_mean_watts', 
                'signal_strength_mean_dbm', 'snr_mean_db', 'noise_level_mean_dbm']
    
    scaler = StandardScaler()
    ref_scaled = scaler.fit_transform(df_ref[features])
    new_scaled = scaler.transform(pd.DataFrame([[freq, bw, power, sig_strength, snr, noise]], columns=features))
    distances = np.linalg.norm(ref_scaled - new_scaled, axis=1)
    
    df_result = df_ref.copy()
    df_result['similarity_%'] = (np.exp(-distances/2) * 100).round(1)
    best = df_result.sort_values('similarity_%', ascending=False).iloc[0]

    st.divider()

    tab_id, tab_radio, tab_traj = st.tabs(["⚓ IDENTITÉ DU NAVIRE", 
                                           "📡 SIGNATURE RADIO", 
                                           "📍 SURVEILLANCE AIS / TRAJECTOIRE"])

    # ====================== IDENTITÉ ======================
    with tab_id:
        st.subheader("FICHE DE RENSEIGNEMENT")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MMSI", int(best['mmsi']))
        c2.metric("Type", best['type'])
        c3.metric("Tonnage", f"{best['gross_tonnage']:,} GT")
        c4.metric("Confiance d'Identification", f"{best['similarity_%']}%")

        st.markdown("### 🏳️ ANALYSE PAVILLONS")
        p1, p2 = st.columns(2)
        p1.info(f"**Pavillon supposé (MMSI)** : {best['pavillon_suppose']}")
        p2.info(f"**Pavillon déclaré** : {best['flag']}")
        
        if best['pavillon_suppose'] != best['flag']:
            st.error("❗ ALERTE : DISCORDANCE DE PAVILLON (Flag Hopping suspect)")
        if best.get('is_suspicious', False):
            st.warning("🚨 NAVIRE SOUS SURVEILLANCE – LISTE NOIRE")

    # ====================== RADIO ======================
    with tab_radio:
        st.subheader("EXPERTISE TECHNIQUE DU SIGNAL")
        tolerances = {'freq_mean_mhz': 0.05, 'bandwidth_mean_khz': 3.0, 'power_mean_watts': 30.0,
                      'signal_strength_mean_dbm': 6.0, 'snr_mean_db': 5.0, 'noise_level_mean_dbm': 8.0}
        
        comp_df = pd.DataFrame({
            'Paramètre': features,
            'Valeur Captée': [round(x, 3) for x in [freq, bw, power, sig_strength, snr, noise]],
            'Valeur Référence': [round(best[f], 3) for f in features],
            'Écart': [round(abs(v - best[f]), 3) for v, f in zip([freq, bw, power, sig_strength, snr, noise], features)],
            'Tolérance': [tolerances[f] for f in features]
        })
        comp_df['Statut'] = comp_df['Écart'] <= comp_df['Tolérance']
        
        st.dataframe(
            comp_df.style
                .map(lambda x: 'color: #c8102e; font-weight: bold;' if x == False else 'color: #006400;', subset=['Statut'])
                .format({'Statut': lambda x: "✅ CONFORME" if x else "🚨 HORS TOLÉRANCE"}),
            use_container_width=True, 
            hide_index=True
        )

    # ====================== TRAJECTOIRE ======================
    with tab_traj:
        st.subheader("ANALYSE COMPORTEMENTALE & TRAJECTOIRE")
        vessel_history = df_ais_all[df_ais_all['mmsi'] == int(best['mmsi'])].sort_values('timestamp').copy()

        if not vessel_history.empty:
            v_limit = VITESSE_MAX_NAVIRE.get(best['type'], 20.0)

            # === Utilisation directe de la colonne 'speed' ===
            if 'speed' in vessel_history.columns:
                vessel_history = vessel_history.rename(columns={'speed': 'speed_knots'})
            else:
                st.error("Colonne 'speed' non trouvée dans les données AIS.")
                vessel_history['speed_knots'] = 0.0

            # Calcul des distances (conservé pour l'alerte de sauts anormaux)
            vessel_history['distance_nm'] = 0.0
            for i in range(1, len(vessel_history)):
                prev = vessel_history.iloc[i-1]
                curr = vessel_history.iloc[i]
                dist = haversine_knots(prev['longitude'], prev['latitude'],
                                       curr['longitude'], curr['latitude'])
                vessel_history.iloc[i, vessel_history.columns.get_loc('distance_nm')] = round(dist, 2)

            # === ALERTES ===
            if (vessel_history['speed_knots'] > v_limit).any():
                st.markdown(f"""
                <div class="big-alert alert-danger">
                    ⚠️ VITESSE MAXIMALE DÉPASSÉE ({(vessel_history['speed_knots'] > v_limit).sum()} positions)
                </div>
                """, unsafe_allow_html=True)

            distances_anormales = vessel_history[vessel_history['distance_nm'] > DISTANCE_ANORMALE_NM]
            if len(distances_anormales) >= 2:
                st.markdown(f"""
                <div class="big-alert alert-warning">
                    ⚠️ {len(distances_anormales)} SAUTS DE DISTANCE ANORMAUX (> {DISTANCE_ANORMALE_NM} NM)
                </div>
                """, unsafe_allow_html=True)

            # Métriques
            colm1, colm2 = st.columns(2)
            with colm1:
                if vessel_history.iloc[-1].get('ais_active', True):
                    st.success("🛰️ AIS ACTIF")
                else:
                    st.error("🌑 AIS INACTIF – DARK ACTIVITY")
            with colm2:
                st.info(f"**Vitesse maximale autorisée** : {v_limit} nœuds")

            # Tableau
            display_hist = vessel_history[['timestamp', 'latitude', 'longitude', 
                                           'speed_knots', 'distance_nm']].copy().iloc[::-1]
            display_hist['Conformité Vitesse'] = display_hist['speed_knots'] <= v_limit

            st.dataframe(
                display_hist.style
                .apply(lambda row: ['background-color: rgba(200, 16, 46, 0.15)']*len(row) 
                       if not row['Conformité Vitesse'] else ['']*len(row), axis=1)
                .format({'speed_knots': "{:.2f} kts", 'distance_nm': "{:.1f} NM"}),
                use_container_width=True, 
                hide_index=True
            )

            # Carte
            fig_map = px.scatter_mapbox(vessel_history, lat="latitude", lon="longitude", 
                                       color="speed_knots", 
                                       hover_data=["timestamp", "speed_knots", "distance_nm"],
                                       zoom=2, height=550, color_continuous_scale="Reds")
            fig_map.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_map, use_container_width=True)

        else:
            st.warning("Aucun historique AIS disponible pour ce MMSI.")