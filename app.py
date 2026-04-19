import streamlit as st
import pandas as pd
import plotly.express as px
import random
from sklearn.neighbors import NearestNeighbors
from transformers import pipeline

# --- DICTIONARIES & SETUP ---
core_features = ['valence', 'energy', 'danceability', 'acousticness', 'tempo']

emotion_baselines = {
    "happy":     {"valence": 0.8, "energy": 0.7, "danceability": 0.7, "acousticness": 0.2, "tempo": 0.6},
    "sad":       {"valence": 0.2, "energy": 0.3, "danceability": 0.1, "acousticness": 0.8, "tempo": 0.3},
    "energetic": {"valence": 0.7, "energy": 0.9, "danceability": 0.8, "acousticness": 0.1, "tempo": 0.8},
    "calm":      {"valence": 0.7, "energy": 0.2, "danceability": 0.2, "acousticness": 0.9, "tempo": 0.3},
    "stressed":  {"valence": 0.2, "energy": 0.8, "danceability": 0.4, "acousticness": 0.1, "tempo": 0.7},
    "angry":     {"valence": 0.1, "energy": 0.9, "danceability": 0.4, "acousticness": 0.0, "tempo": 0.8},
    "romantic":  {"valence": 0.8, "energy": 0.4, "danceability": 0.5, "acousticness": 0.7, "tempo": 0.4},
    "chill":     {"valence": 0.6, "energy": 0.4, "danceability": 0.6, "acousticness": 0.5, "tempo": 0.4},
    "focus":     {"valence": 0.5, "energy": 0.3, "danceability": 0.2, "acousticness": 0.8, "tempo": 0.4},
    "party":     {"valence": 0.8, "energy": 0.9, "danceability": 0.9, "acousticness": 0.1, "tempo": 0.8},
    "tired":     {"valence": 0.4, "energy": 0.2, "danceability": 0.2, "acousticness": 0.8, "tempo": 0.3},
    "motivated": {"valence": 0.7, "energy": 0.8, "danceability": 0.6, "acousticness": 0.1, "tempo": 0.7},
    "bored":     {"valence": 0.4, "energy": 0.4, "danceability": 0.5, "acousticness": 0.4, "tempo": 0.5},
    "nostalgic": {"valence": 0.5, "energy": 0.4, "danceability": 0.4, "acousticness": 0.7, "tempo": 0.4},
    "frustrated":{"valence": 0.3, "energy": 0.7, "danceability": 0.3, "acousticness": 0.1, "tempo": 0.6}
}

hashtag_modifiers = {
    "#WorkMode": {"energy": -0.2, "acousticness": 0.3, "danceability": -0.2},
    "#PartyMode": {"danceability": 0.4, "energy": 0.3, "valence": 0.2},
    "#Acoustic": {"acousticness": 0.4, "energy": -0.2},
    "#BrainFocus": {"energy": -0.2, "acousticness": 0.3, "danceability": -0.2},
    "#Groove": {"danceability": 0.3, "energy": 0.2, "tempo": 0.1},
    "#Workout": {"energy": 0.4, "tempo": 0.2, "danceability": 0.2},
    "#SpringDays": {"valence": 0.4, "energy": 0.2, "acousticness": 0.2},
    "#GloomyDays": {"valence": -0.3, "energy": -0.2, "acousticness": 0.3},
    "#LateNightDrive": {"energy": -0.3, "danceability": 0.2, "tempo": -0.2},
    "#MainCharacter": {"valence": 0.3, "energy": 0.3, "tempo": 0.1},
    "#RoadTrip": {"energy": 0.2, "valence": 0.2, "danceability": 0.1},
    "#LoFiSleep": {"energy": -0.4, "acousticness": 0.4, "tempo": -0.3, "danceability": -0.2},
    "#SundayMorning": {"acousticness": 0.4, "energy": -0.2, "valence": 0.2},
}

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI VibeMatch", page_icon="🎧", layout="centered")

# --- CACHING DATA & AI MODELS ---
@st.cache_data
def load_data():
    return pd.read_csv("processed_spotify_data.csv")

@st.cache_resource
def load_nlp_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# HEADER
st.title("🎧 AI VibeMatch")
st.subheader("Context-Aware Music Recommendation Engine")
st.write("Tell us how you're feeling for the day, and our Machine Learning engine will curate the perfect 30-track playlist for you <3")
st.divider()
with st.spinner("Initializing AI Engines (this might take a moment on the first load)..."):
    df_clean = load_data()
    classifier = load_nlp_model()


# --- SESSION STATE FOR RANDOM GENRES ---
all_genres = df_clean['track_genre'].unique().tolist()
if 'random_genres' not in st.session_state:
    st.session_state.random_genres = random.sample(all_genres, 7)
if 'random_hashtags' not in st.session_state:
    st.session_state.random_hashtags = random.sample(list(hashtag_modifiers.keys()), 5)
if 'show_results' not in st.session_state:
    st.session_state.show_results = False

# --- USER INPUTS ---
if not st.session_state.show_results:
    # DIARY ENTRY UI
    st.write("### 1️⃣ What's on your mind?📝")
    user_text = st.text_area(
        "Write a short diary entry, or just describe how you are feeling right now: \n\n(Press Ctrl+Enter to submit)",
        placeholder="I have a huge economics exam tomorrow, I'm so overwhelmed and tired...")

    # THE HASHTAG SELECTOR UI
    st.write("### 2️⃣ Pick a hashtag for the moment✨")
    vibe_options = st.session_state.random_hashtags + ["Skip..."]
    selected_vibe = st.pills("Select a hashtag (or Skip):", options=vibe_options, selection_mode="single", label_visibility="collapsed")
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Refresh Tags"):
            st.session_state.random_hashtags = random.sample(list(hashtag_modifiers.keys()), 5)
            st.rerun()

    # THE GENRE SELECTOR UI
    st.write("### 3️⃣ Pick one or more genres🪩🎸")
    display_genres = [g.title() for g in st.session_state.random_genres]
    genre_options = display_genres + ["Skip..."]
    selected_genres_display = st.pills("Select genres (or Skip):", options=genre_options, selection_mode="multi", label_visibility="collapsed")
    col3, col4 = st.columns([1, 4])
    with col3:
        if st.button("🔄 Different genres"):
            st.session_state.random_genres = random.sample(all_genres, 7)
            st.rerun()
    st.divider()

# --- THE GENERATE BUTTON ---
    if st.button("💽 Generate My Playlist"):
        if not user_text.strip():
            st.warning("Please write a little bit about how you're feeling in 1️⃣!")
        else:
            with st.spinner("Analyzing your text and calculating audio vectors..."):
                # Run NLP
                candidate_labels = list(emotion_baselines.keys())
                result = classifier(user_text, candidate_labels)
                st.session_state.top_emotion = result['labels'][0]

                # Format genres back to lowercase for the dataframe math
                if selected_genres_display is None:
                    selected_genres_display = []
                final_genres = [g.lower() for g in selected_genres_display if g != "Skip..."]

                # Math
                target_features = emotion_baselines[st.session_state.top_emotion].copy()
                if selected_vibe and selected_vibe != "Skip..." and selected_vibe in hashtag_modifiers:
                    for feature, change_value in hashtag_modifiers[selected_vibe].items():
                        new_value = target_features[feature] + change_value
                        target_features[feature] = max(0.0, min(1.0, new_value))

                st.session_state.target_features = target_features

                target_array = pd.DataFrame([[
                    target_features['valence'], target_features['energy'], target_features['danceability'],
                    target_features['acousticness'], target_features['tempo']
                ]], columns=core_features)

                # KNN Model
                if len(final_genres) > 0:
                    genre_matches = df_clean[df_clean['track_genre'].isin(final_genres)]
                    if len(genre_matches) >= 30:
                        knn = NearestNeighbors(n_neighbors=30, metric='euclidean').fit(genre_matches[core_features])
                        _, indices = knn.kneighbors(target_array)
                        st.session_state.playlist = genre_matches.iloc[indices[0]]
                    else:
                        st.session_state.playlist = genre_matches # Simplified fallback
                else:
                    knn = NearestNeighbors(n_neighbors=30, metric='euclidean').fit(df_clean[core_features])
                    _, indices = knn.kneighbors(target_array)
                    st.session_state.playlist = df_clean.iloc[indices[0]]

                # Switch views!
                st.session_state.show_results = True
                st.rerun()
# --- DISPLAY RESULTS ---
else:
    st.success("✨ Playlist Generated Successfully!")
    st.balloons()

    st.info(f"🧠 **AI Text Analysis Complete!** We detected your core emotion as: **{st.session_state.top_emotion.capitalize()}**")

    # Plotly Radar Chart
    radar_data = pd.DataFrame(dict(
        r=[st.session_state.target_features[f] for f in core_features],
        theta=[f.capitalize() for f in core_features],
        text=[f"{st.session_state.target_features[f]:.2f}" for f in core_features]
    ))
    fig = px.line_polar(radar_data, r='r', theta='theta', line_close=True, range_r=[0, 1], text='text')
    fig.update_traces(fill='toself', line_color='#1DB954', textposition='top center')
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350)

    st.plotly_chart(fig)

    # Display Playlist
    st.write("### Your Curated Tracks")
    display_df = st.session_state.playlist[['track_name', 'artists', 'album_name', 'track_genre']].copy()
    display_df.columns = ["Track Name", "Artist", "Album", "Genre"]
    display_df['Genre'] = display_df['Genre'].str.title()
    display_df.index = range(1, len(display_df) + 1)

    st.dataframe(display_df)

    st.divider()
    if st.button("⬅️ Start Over"):
        st.session_state.show_results = False
        st.rerun()