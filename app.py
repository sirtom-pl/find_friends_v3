import json
import streamlit as st
import pandas as pd  # type: ignore
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore

MODEL_NAME = 'welcome_survey_clustering_pipeline_v1'
DATA = 'welcome_survey_simple_v1.csv'
CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v1.json'
COLOR_PALETTE = px.colors.qualitative.Pastel


@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants():
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)
    return df_with_clusters

with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    age = st.selectbox("Wiek", ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])

    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender,
        }
    ])

model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data['description'])
same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
st.metric("Liczba twoich znajomych", len(same_cluster_df))

st.header("Osoby z grupy")

# Wiek
fig = px.histogram(same_cluster_df.sort_values("age"), x="age", color="age", color_discrete_sequence=COLOR_PALETTE)
fig.update_layout(
    title="Rozkład wieku w grupie",
    xaxis_title="Wiek",
    yaxis_title="Liczba osób",
    font=dict(color="black"),
    xaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
    yaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
    showlegend=False
)
st.plotly_chart(fig)

# Wykształcenie
fig = px.histogram(same_cluster_df, x="edu_level", color="edu_level", color_discrete_sequence=COLOR_PALETTE)
fig.update_layout(
    title="Rozkład wykształcenia w grupie",
    xaxis_title="Wykształcenie",
    yaxis_title="Liczba osób",
    font=dict(color="black"),
    xaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
    yaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
    showlegend=False
)
st.plotly_chart(fig)

# Ulubione zwierzęta
fig = px.histogram(same_cluster_df, x="fav_animals", color="fav_animals", color_discrete_sequence=COLOR_PALETTE)
fig.update_layout(
    title="Rozkład ulubionych zwierząt w grupie",
    xaxis_title="Ulubione zwierzęta",
    yaxis_title="Liczba osób",
    font=dict(color="black"),
    xaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
    yaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
    showlegend=False
)
st.plotly_chart(fig)

# Ulubione miejsce
fig = px.histogram(same_cluster_df, x="fav_place", color="fav_place", color_discrete_sequence=COLOR_PALETTE)
fig.update_layout(
    title="Rozkład ulubionych miejsc w grupie",
    xaxis_title="Ulubione miejsce",
    yaxis_title="Liczba osób",
    font=dict(color="black"),
    xaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
    yaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
    showlegend=False
)
st.plotly_chart(fig)

# Płeć
fig = px.histogram(same_cluster_df, x="gender", color="gender", color_discrete_sequence=COLOR_PALETTE)
fig.update_layout(
    title="Rozkład płci w grupie",
    xaxis_title="Płeć",
    yaxis_title="Liczba osób",
    font=dict(color="black"),
    xaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
    yaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
    showlegend=False
)
st.plotly_chart(fig)

