import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import plotly.express as px
from datetime import datetime, timedelta
import sqlite3
from langdetect import detect
import unittest
import os
import json  # For JSON export

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Set up the page configuration for a professional layout
st.set_page_config(page_title="Customer Feedback Analysis Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Load pre-trained models for emotion detection with error handling
try:
    emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base",
                                  top_k=None)
    tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# Define emotion activation levels (matching your specified lists)
emotion_activation = {
    'ecstasy': 'High', 'vigilance': 'High', 'admiration': 'High', 'terror': 'High',
    'amazement': 'High', 'grief': 'High', 'loathing': 'High', 'rage': 'High',
    'joy': 'Medium', 'anticipation': 'Medium', 'trust': 'Medium', 'fear': 'Medium',
    'surprise': 'Medium', 'sadness': 'Medium', 'anger': 'Medium', 'disgust': 'Medium',
    'serenity': 'Low', 'interest': 'Low', 'acceptance': 'Low', 'apprehension': 'Low',
    'distraction': 'Low', 'pensiveness': 'Low', 'boredom': 'Low', 'annoyance': 'Low'
}


# Database functions for persistent storage (only text and date)
def save_feedback(text, date):
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS feedback (text TEXT, date TEXT)')
    c.execute('INSERT INTO feedback VALUES (?, ?)', (text, date.isoformat()))
    conn.commit()
    conn.close()


def load_feedback():
    try:
        conn = sqlite3.connect('feedback.db')
        df = pd.read_sql_query('SELECT * FROM feedback', conn)
        conn.close()
        return [{'text': row['text'], 'date': datetime.fromisoformat(row['date'])} for _, row in df.iterrows()]
    except Exception as e:
        st.warning(f"No feedback data found or error loading: {str(e)}")
        return []


# Function to detect emotions and their activation/intensity
@st.cache_data
def analyze_emotions(text):
    try:
        if not text.strip():
            raise ValueError("Empty feedback provided")
        language = detect(text)
        if language != 'en':
            st.warning("Non-English feedback detected. Analysis may be less accurate.")
        result = emotion_classifier(text)[0]
        emotions = {item['label']: item['score'] for item in result}

        emotion_mapping = {emotion: {'activation': emotion_activation[emotion], 'intensity': score}
                           for emotion, score in emotions.items() if emotion in emotion_activation}

        sorted_emotions = sorted([(e, s) for e, s in emotions.items() if e in emotion_activation], key=lambda x: x[1],
                                 reverse=True)
        primary_emotion = sorted_emotions[0][0] if sorted_emotions else 'neutral'
        secondary_emotion = sorted_emotions[1][0] if len(sorted_emotions) > 1 else 'neutral'

        return {
            'primary': {
                'emotion': primary_emotion,
                'activation': emotion_mapping[primary_emotion]['activation'],
                'intensity': emotion_mapping[primary_emotion]['intensity']
            },
            'secondary': {
                'emotion': secondary_emotion,
                'activation': emotion_mapping[secondary_emotion]['activation'],
                'intensity': emotion_mapping[secondary_emotion]['intensity']
            }
        }
    except Exception as e:
        st.error(f"Error analyzing emotions: {str(e)}")
        return {'primary': {'emotion': 'neutral', 'activation': 'Low', 'intensity': 0.0},
                'secondary': {'emotion': 'neutral', 'activation': 'Low', 'intensity': 0.0}}


# Function for topic analysis (keyword-based)
@st.cache_data
def analyze_topics(text):
    try:
        if not text.strip():
            raise ValueError("Empty feedback provided")
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

        main_topics = ['Delivery', 'Quality', 'Clothes']
        subtopics = {
            'Delivery': ['fast delivery', 'quick delivery', 'free delivery', 'good delivery'],
            'Quality': ['material quality', 'durability', 'craftsmanship'],
            'Clothes': ['fit', 'style', 'comfort']
        }

        topics_found = {}
        for main_topic in main_topics:
            topics_found[main_topic] = []
            for subtopic in subtopics[main_topic]:
                if subtopic in text.lower():
                    topics_found[main_topic].append(subtopic)

        topic_contributions = {}
        for main_topic in main_topics:
            topic_contributions[main_topic] = any(subtopic in text.lower() for subtopic in subtopics[main_topic])

        return {
            'main': [t for t, v in topic_contributions.items() if v],
            'subtopics': {k: v for k, v in topics_found.items() if v}
        }
    except Exception as e:
        st.error(f"Error analyzing topics: {str(e)}")
        return {'main': [], 'subtopics': {}}


# Function to calculate Adorescore
@st.cache_data
def calculate_adorescore(emotions, topics):
    try:
        emotion_weights = {
            'ecstasy': 0.4, 'vigilance': 0.4, 'admiration': 0.3, 'terror': -0.5,
            'amazement': 0.3, 'grief': -0.4, 'loathing': -0.5, 'rage': -0.5,
            'joy': 0.4, 'anticipation': 0.2, 'trust': 0.3, 'fear': -0.3,
            'surprise': 0.2, 'sadness': -0.4, 'anger': -0.5, 'disgust': -0.3,
            'serenity': 0.2, 'interest': 0.1, 'acceptance': 0.2, 'apprehension': -0.2,
            'distraction': -0.1, 'pensiveness': -0.1, 'boredom': -0.1, 'annoyance': -0.2
        }

        topic_weights = {
            'Delivery': 0.3, 'Quality': 0.4, 'Clothes': 0.3
        }

        emotion_score = 0
        for emotion_type in ['primary', 'secondary']:
            emotion = emotions[emotion_type]['emotion']
            intensity = emotions[emotion_type]['intensity']
            if emotion in emotion_weights:
                emotion_score += emotion_weights[emotion] * intensity

        topic_score = 0
        for topic in topics['main']:
            if topic in topic_weights:
                topic_score += topic_weights[topic] * 0.5

        total_score = (emotion_score + topic_score) * 100
        total_score = max(-100, min(100, total_score))

        breakdown = {}
        for topic in topics['main']:
            breakdown[topic] = int(total_score * topic_weights.get(topic, 0))

        return {'overall': total_score, 'breakdown': breakdown}
    except Exception as e:
        st.error(f"Error calculating Adorescore: {str(e)}")
        return {'overall': 0, 'breakdown': {}}


# Function to process all feedback in the dataset
@st.cache_data
def process_feedback_dataset(feedback_data):
    try:
        all_emotions = []
        all_topics = []
        all_adorescores = []

        for entry in feedback_data:
            emotions = analyze_emotions(entry['text'])
            topics = analyze_topics(entry['text'])
            adorescore = calculate_adorescore(emotions, topics)
            all_emotions.append(emotions)
            all_topics.append(topics)
            all_adorescores.append(adorescore)

        return all_emotions, all_topics, all_adorescores
    except Exception as e:
        st.error(f"Error processing dataset: {str(e)}")
        return [], [], []


# Function to aggregate emotions by activation level for dynamic radar charts
@st.cache_data
def get_emotion_distribution(emotions_list):
    try:
        emotion_intensities = {
            'ecstasy': 0, 'vigilance': 0, 'admiration': 0, 'terror': 0, 'amazement': 0,
            'grief': 0, 'loathing': 0, 'rage': 0, 'joy': 0, 'anticipation': 0, 'trust': 0,
            'fear': 0, 'surprise': 0, 'sadness': 0, 'anger': 0, 'disgust': 0, 'serenity': 0,
            'interest': 0, 'acceptance': 0, 'apprehension': 0, 'distraction': 0, 'pensiveness': 0,
            'boredom': 0, 'annoyance': 0
        }

        for emotions in emotions_list:
            for emotion_type in ['primary', 'secondary']:
                emotion = emotions[emotion_type]['emotion']
                intensity = emotions[emotion_type]['intensity']
                if emotion in emotion_intensities:
                    emotion_intensities[emotion] = emotion_intensities.get(emotion, 0) + intensity

        # Normalize intensities by number of entries
        num_entries = len(emotions_list)
        for emotion in emotion_intensities:
            emotion_intensities[emotion] /= num_entries if num_entries > 0 else 1

        # Group by activation level
        high_emotions = {e: i for e, i in emotion_intensities.items() if emotion_activation[e] == 'High'}
        medium_emotions = {e: i for e, i in emotion_intensities.items() if emotion_activation[e] == 'Medium'}
        low_emotions = {e: i for e, i in emotion_intensities.items() if emotion_activation[e] == 'Low'}

        return high_emotions, medium_emotions, low_emotions
    except Exception as e:
        st.error(f"Error generating emotion distribution: {str(e)}")
        return {}, {}, {}


# Load or initialize feedback data
sample_feedback_data = load_feedback() or [
    {
        "text": "The delivery was incredibly fast and the quality was amazing! However, one of the clothing items didn't fit well.",
        "date": datetime.now() - timedelta(days=5)},
    {"text": "Great service, fast delivery, but the material quality is poor.",
     "date": datetime.now() - timedelta(days=3)},
    {"text": "Love the clothes, perfect fit, but delivery took too long.", "date": datetime.now() - timedelta(days=1)}
]

# User authentication
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("Login")
    st.divider()  # Add divider for professional look
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login", use_container_width=True):
        if username == "admin" and password == "password123":  # Replace with secure auth in production
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid credentials")
else:
    st.title("Customer Feedback Analysis Dashboard")
    st.divider()  # Add divider for professional look

    # Sidebar for filters
    st.sidebar.header("Filters")
    time_range = st.sidebar.selectbox("Time Range", ["Last 30 days", "Last 7 days", "All Time"])
    emotion_filter = st.sidebar.selectbox("Show Analysis for:",
                                          ["Joy", "Sadness", "Anger", "Fear", "Disgust", "Surprise", "Neutral"])
    adorescore_range = st.sidebar.slider("Adorescore Range", -100, 100, (-100, 100))
    topic_filter = st.sidebar.multiselect("Filter by Topic", ['Delivery', 'Quality', 'Clothes'])
    search_query = st.sidebar.text_input("Search Feedback", "")

    # Sample data filtering based on time range and advanced filters
    if time_range == "Last 30 days":
        filtered_data = [entry for entry in sample_feedback_data if (datetime.now() - entry['date']).days <= 30]
    elif time_range == "Last 7 days":
        filtered_data = [entry for entry in sample_feedback_data if (datetime.now() - entry['date']).days <= 7]
    else:
        filtered_data = sample_feedback_data

    # Apply advanced filters
    filtered_data_with_analysis = []
    for entry in filtered_data:
        emotions = analyze_emotions(entry['text'])
        topics = analyze_topics(entry['text'])
        adorescore = calculate_adorescore(emotions, topics)
        if topic_filter and not any(t in topics['main'] for t in topic_filter):
            continue
        if search_query and search_query.lower() not in entry['text'].lower():
            continue
        if not (adorescore_range[0] <= adorescore['overall'] <= adorescore_range[1]):
            continue
        filtered_data_with_analysis.append(
            {'entry': entry, 'emotions': emotions, 'topics': topics, 'adorescore': adorescore})

    # Process the filtered dataset
    try:
        all_emotions = [item['emotions'] for item in filtered_data_with_analysis]
        all_topics = [item['topics'] for item in filtered_data_with_analysis]
        all_adorescores = [item['adorescore'] for item in filtered_data_with_analysis]
    except Exception as e:
        st.error(f"Error processing dataset: {str(e)}")
        all_emotions, all_topics, all_adorescores = [], [], []

    # Input area for new feedback
    st.subheader("Enter New Customer Feedback", divider="gray")
    new_feedback = st.text_area("Feedback:",
                                "The delivery was incredibly fast and the quality was amazing! However, one of the clothing items didn't fit well.")

    if st.button("Analyze New Feedback", use_container_width=True):
        if not new_feedback.strip():
            st.error("Please enter feedback before analyzing.")
        else:
            try:
                emotions = analyze_emotions(new_feedback)
                topics = analyze_topics(new_feedback)
                adorescore = calculate_adorescore(emotions, topics)

                # Update sample data with new feedback
                new_entry = {"text": new_feedback, "date": datetime.now()}
                sample_feedback_data.append(new_entry)
                save_feedback(new_feedback, datetime.now())

                # Re-process dataset
                filtered_data_with_analysis = []
                for entry in sample_feedback_data:
                    emotions = analyze_emotions(entry['text'])
                    topics = analyze_topics(entry['text'])
                    adorescore = calculate_adorescore(emotions, topics)
                    filtered_data_with_analysis.append(
                        {'entry': entry, 'emotions': emotions, 'topics': topics, 'adorescore': adorescore})

                all_emotions = [item['emotions'] for item in filtered_data_with_analysis]
                all_topics = [item['topics'] for item in filtered_data_with_analysis]
                all_adorescores = [item['adorescore'] for item in filtered_data_with_analysis]

                # Display results
                st.subheader("New Feedback Analysis", divider="gray")
                st.markdown("<p style='color: white;'>Emotions:</p>", unsafe_allow_html=True)
                st.markdown(
                    f"<p style='color: white;'>Primary: {emotions['primary']['emotion']} (Activation: {emotions['primary']['activation']}, Intensity: {emotions['primary']['intensity']:.4f})</p>",
                    unsafe_allow_html=True)
                st.markdown(
                    f"<p style='color: white;'>Secondary: {emotions['secondary']['emotion']} (Activation: {emotions['secondary']['activation']}, Intensity: {emotions['secondary']['intensity']:.4f})</p>",
                    unsafe_allow_html=True)
                st.markdown("<p style='color: white;'>Topics:</p>", unsafe_allow_html=True)
                for main_topic, subtopics in topics['subtopics'].items():
                    st.markdown(
                        f"<p style='color: white;'>{main_topic}: {', '.join(subtopics) if subtopics else 'None'}</p>",
                        unsafe_allow_html=True)
                st.markdown("<p style='color: white;'>Adorescore:</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: white;'>Overall: {adorescore['overall']:.2f}</p>",
                            unsafe_allow_html=True)
                st.markdown("<p style='color: white;'>Breakdown:</p>", unsafe_allow_html=True)
                for topic, score in adorescore['breakdown'].items():
                    st.markdown(f"<p style='color: white;'>{topic}: {score}</p>", unsafe_allow_html=True)

                # Recommendations
                if adorescore['breakdown'].get('Delivery', 0) < 50:
                    st.warning("Consider improving delivery speed or reliability.")
                if adorescore['breakdown'].get('Quality', 0) < 50:
                    st.warning("Consider improving product quality, such as material or durability.")
                if adorescore['breakdown'].get('Clothes', 0) < 50:
                    st.warning("Consider improving clothing fit or style.")
            except Exception as e:
                st.error(f"Error analyzing feedback: {str(e)}")

    # Emotion Analysis Section (professional layout with neatly aligned labels and buttons)
    st.header("Emotion Analysis", divider="gray")
    st.caption("Analyzing customer emotions and sentiments across different themes. [:link: Share] [:download: Export]")

    # Get dynamic emotion distributions
    high_emotions, medium_emotions, low_emotions = get_emotion_distribution(all_emotions)

    col1, col2, col3, col4 = st.columns(4, gap="medium")

    with col1:
        st.subheader("HIGH ACTIVATION EMOTIONS")
        # Find the highest-intensity emotion in high-activation category
        high_emotion_label = max(high_emotions.items(), key=lambda x: x[1])[0] if high_emotions else 'Ecstasy'
        st.button(high_emotion_label, key="high_activation",
                  help="Highest intensity emotion in High Activation")  # Removed 'style' parameter
        # Create dynamic radar chart with neatly aligned labels and filled area
        high_emotions_list = ['ecstasy', 'vigilance', 'admiration', 'terror', 'amazement', 'grief', 'loathing', 'rage']
        high_intensities = [high_emotions.get(emotion, 0) for emotion in high_emotions_list]
        fig_high = px.line_polar(
            r=high_intensities,
            theta=high_emotions_list,
            line_close=True,
            hover_data={'Intensity': high_intensities},
            title="High Activation Emotions"
        )
        # Set line color, width, and add filled area manually using update_traces
        fig_high.update_traces(line_color="gray", line_width=2, fill='toself',
                               mode='lines')  # Use 'lines' mode for line_polar and fill
        fig_high.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    showticklabels=True,
                    tickfont=dict(size=12, color="black"),
                    gridcolor="lightgray"
                ),
                angularaxis=dict(
                    tickfont=dict(size=12, color="black"),
                    rotation=90,  # Start labels at top for neat alignment
                    direction="clockwise",
                    showticklabels=True,
                    tickangle=45  # Adjust label angle for better readability
                )
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=False,
            title_font=dict(size=16, color="black"),
            title_x=0.5,  # Center the title
            margin=dict(l=20, r=20, t=40, b=20)  # Adjust margins for better spacing
        )
        st.plotly_chart(fig_high, use_container_width=True)

    with col2:
        st.subheader("MEDIUM ACTIVATION EMOTIONS")
        # Find the highest-intensity emotion in medium-activation category
        medium_emotion_label = max(medium_emotions.items(), key=lambda x: x[1])[0] if medium_emotions else 'Joy'
        st.button(medium_emotion_label, key="medium_activation",
                  help="Highest intensity emotion in Medium Activation")  # Removed 'style' parameter
        # Create dynamic radar chart with neatly aligned labels and filled area
        medium_emotions_list = ['joy', 'anticipation', 'trust', 'fear', 'surprise', 'sadness', 'anger', 'disgust']
        medium_intensities = [medium_emotions.get(emotion, 0) for emotion in medium_emotions_list]
        fig_medium = px.line_polar(
            r=medium_intensities,
            theta=medium_emotions_list,
            line_close=True,
            hover_data={'Intensity': medium_intensities},
            title="Medium Activation Emotions"
        )
        # Set line color, width, and add filled area manually using update_traces
        fig_medium.update_traces(line_color="gray", line_width=2, fill='toself',
                                 mode='lines')  # Use 'lines' mode for line_polar and fill
        fig_medium.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    showticklabels=True,
                    tickfont=dict(size=12, color="black"),
                    gridcolor="lightgray"
                ),
                angularaxis=dict(
                    tickfont=dict(size=12, color="black"),
                    rotation=90,  # Start labels at top for neat alignment
                    direction="clockwise",
                    showticklabels=True,
                    tickangle=45  # Adjust label angle for better readability
                )
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=False,
            title_font=dict(size=16, color="black"),
            title_x=0.5,  # Center the title
            margin=dict(l=20, r=20, t=40, b=20)  # Adjust margins for better spacing
        )
        st.plotly_chart(fig_medium, use_container_width=True)

    with col3:
        st.subheader("LOW ACTIVATION EMOTIONS")
        # Find the highest-intensity emotion in low-activation category
        low_emotion_label = max(low_emotions.items(), key=lambda x: x[1])[0] if low_emotions else 'Serenity'
        st.button(low_emotion_label, key="low_activation",
                  help="Highest intensity emotion in Low Activation")  # Removed 'style' parameter
        # Create dynamic radar chart with neatly aligned labels and filled area
        low_emotions_list = ['serenity', 'interest', 'acceptance', 'apprehension', 'distraction', 'pensiveness',
                             'boredom', 'annoyance']
        low_intensities = [low_emotions.get(emotion, 0) for emotion in low_emotions_list]
        fig_low = px.line_polar(
            r=low_intensities,
            theta=low_emotions_list,
            line_close=True,
            hover_data={'Intensity': low_intensities},
            title="Low Activation Emotions"
        )
        # Set line color, width, and add filled area manually using update_traces
        fig_low.update_traces(line_color="gray", line_width=2, fill='toself',
                              mode='lines')  # Use 'lines' mode for line_polar and fill
        fig_low.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    showticklabels=True,
                    tickfont=dict(size=12, color="black"),
                    gridcolor="lightgray"
                ),
                angularaxis=dict(
                    tickfont=dict(size=12, color="black"),
                    rotation=90,  # Start labels at top for neat alignment
                    direction="clockwise",
                    showticklabels=True,
                    tickangle=45  # Adjust label angle for better readability
                )
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=False,
            title_font=dict(size=16, color="black"),
            title_x=0.5,  # Center the title
            margin=dict(l=20, r=20, t=40, b=20)  # Adjust margins for better spacing
        )
        st.plotly_chart(fig_low, use_container_width=True)

    with col4:
        st.subheader("Adorescore", divider="gray")
        overall_adorescore = np.mean([score['overall'] for score in all_adorescores]) if all_adorescores else 0
        st.markdown(f"<h4 style='color: white; text-align: center;'>+{int(overall_adorescore):.2f}</h4>",
                    unsafe_allow_html=True)
        all_emotion_intensities = {}
        for emotions in all_emotions:
            for emotion_type in ['primary', 'secondary']:
                emotion = emotions[emotion_type]['emotion']
                intensity = emotions[emotion_type]['intensity']
                all_emotion_intensities[emotion] = all_emotion_intensities.get(emotion, 0) + intensity
        driving_emotion = max(all_emotion_intensities.items(), key=lambda x: x[1])[
            0] if all_emotion_intensities else 'Joy'
        st.markdown(f"<p style='color: white; text-align: center;'>Driven by</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='color: white; text-align: center;'>{driving_emotion} - {int(np.mean(list(all_emotion_intensities.values())) * 100)}%</p>",
            unsafe_allow_html=True)
        st.markdown(f"<p style='color: white; text-align: center;'>Top Themes in Dataset</p>", unsafe_allow_html=True)
        theme_scores = {}
        for topics in all_topics:
            for topic in topics['main']:
                theme_scores[topic] = theme_scores.get(topic, 0) + 1
        total_entries = len(all_topics)
        for theme in ['Delivery', 'Quality', 'Clothes']:
            score = int((theme_scores.get(theme, 0) / total_entries) * 100) if total_entries > 0 else 0
            st.markdown(f"<p style='color: white; text-align: center;'>{theme}: {score}</p>", unsafe_allow_html=True)

    # Export functionality (JSON format)
    if st.button("Export Data (JSON)", use_container_width=True):
        export_data = []
        for item in filtered_data_with_analysis:
            entry_data = {
                "emotions": {
                    "primary": {
                        "emotion": item['emotions']['primary']['emotion'],
                        "activation": item['emotions']['primary']['activation'],
                        "intensity": item['emotions']['primary']['intensity']
                    },
                    "secondary": {
                        "emotion": item['emotions']['secondary']['emotion'],
                        "activation": item['emotions']['secondary']['activation'],
                        "intensity": item['emotions']['secondary']['intensity']
                    }
                },
                "topics": {
                    "main": item['topics']['main'],
                    "subtopics": item['topics']['subtopics']
                },
                "adorescore": {
                    "overall": item['adorescore']['overall'],
                    "breakdown": item['adorescore']['breakdown']
                }
            }
            export_data.append(entry_data)

        # Convert to JSON string
        json_data = json.dumps(export_data, indent=4)
        st.download_button("Download JSON", json_data, "feedback_export.json", "application/json")

    # Themes Section with Filters (professional layout)
    st.header("Themes", divider="gray")
    st.caption(f"Showing analysis for: {emotion_filter} {int(overall_adorescore)} in CUSTOMERS ASOS 34")
    st.button("Advanced Filters", use_container_width=True, key="advanced_filters")

    # Theme analysis
    col5, col6 = st.columns(2, gap="medium")

    with col5:
        st.markdown("<h4 style='color: white;'>Search themes...</h4>", unsafe_allow_html=True)
        for theme in ['Delivery', 'Quality', 'Clothes']:
            checked = st.checkbox(theme, value=True if theme == 'Delivery' else False)
            if checked:
                volume = np.random.randint(5, 10)  # Simulate volume
                score = theme_scores.get(theme, 0)
                st.markdown(f"<p style='color: white; text-align: center;'>Score: {score} · Volume: {volume}%</p>",
                            unsafe_allow_html=True)

    with col6:
        st.markdown("<h4 style='color: white;'>Filter by Subtopic</h4>", unsafe_allow_html=True)
        subtopics = {
            'Delivery': ['Fast Delivery', 'Quick Delivery', 'Free Delivery', 'Good Delivery'],
            'Quality': ['Material Quality', 'Durability', 'Craftsmanship'],
            'Clothes': ['Fit', 'Style', 'Comfort']
        }
        for theme in ['Delivery', 'Quality', 'Clothes']:
            if st.checkbox(theme, value=True if theme == 'Delivery' else False, key=f"sub_{theme}"):
                for subtopic in subtopics[theme]:
                    if st.checkbox(subtopic, key=f"sub_{subtopic}"):
                        st.markdown(f"<p style='color: white; text-align: center;'>✓ {subtopic}</p>",
                                    unsafe_allow_html=True)

    # Snippets (Sample Feedback, professional layout)
    st.subheader("Snippets (2)", divider="gray")
    for entry in filtered_data[:2]:  # Show only 2 snippets for demo
        st.markdown(f"<p style='color: white; text-align: center;'>{entry['text']}</p>", unsafe_allow_html=True)

    # Logout button (professional layout)
    if st.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

    # Global styling for professional, clean dashboard
    st.markdown("""
        <style>
        body, .stApp {
            background-color: black;
            color: white;
            font-family: Arial, sans-serif;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stCheckbox>div {
            margin: 5px 0;
        }
        .stHeader {
            font-size: 24px;
            font-weight: bold;
            color: white;
            margin-bottom: 10px;
        }
        .stSubheader {
            color: white;
            margin-bottom: 5px;
        }
        .stCaption {
            color: gray;
            margin-bottom: 10px;
        }
        /* Style for emotion buttons in Emotion Analysis */
        .stButton>button[key="high_activation"],
        .stButton>button[key="medium_activation"],
        .stButton>button[key="low_activation"] {
            background-color: #808080 !important; /* Gray background */
            color: white !important;
            padding: 5px 10px !important;
            border: none !important;
            border-radius: 15px !important;
            margin-bottom: 10px !important;
        }
        /* Override Emotion Analysis section background to white */
        .element-container:nth-child(7) .stColumn > div {
            background-color: white !important;
            color: black !important;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .element-container:nth-child(7) .stColumn h1, 
        .element-container:nth-child(7) .stColumn h4, 
        .element-container:nth-child(7) .stColumn p {
            color: black !important;
        }
        /* Add dividers and spacing for professional look */
        .stDivider {
            margin: 20px 0;
        }
        .stColumn {
            padding: 10px;
        }
        </style>
    """, unsafe_allow_html=True)


# Unit tests (for development, not run in production)
class TestDashboard(unittest.TestCase):
    def test_emotion_analysis(self):
        text = "I love the fast delivery!"
        emotions = analyze_emotions(text)
        self.assertIn('joy', emotions['primary']['emotion'])

    def test_topic_analysis(self):
        text = "Fast delivery and great quality."
        topics = analyze_topics(text)
        self.assertIn('Delivery', topics['main'])
        self.assertIn('Quality', topics['main'])

    def test_adorescore(self):
        emotions = {'primary': {'emotion': 'joy', 'activation': 'Medium', 'intensity': 0.9},
                    'secondary': {'emotion': 'neutral', 'activation': 'Low', 'intensity': 0.1}}
        topics = {'main': ['Delivery', 'Quality'], 'subtopics': {'Delivery': ['fast delivery']}}
        adorescore = calculate_adorescore(emotions, topics)
        self.assertTrue(-100 <= adorescore['overall'] <= 100)


if __name__ == '__main__':
    # Run tests if in development mode (optional)
    if os.environ.get('TEST_MODE', 'False') == 'True':
        unittest.main(argv=['first-arg-is-ignored'], exit=False)
    else:
        # Ensure Streamlit runs the app
        pass
