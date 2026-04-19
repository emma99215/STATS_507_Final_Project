# 🎧 AI VibeMatch
**Context-Aware Music Recommendation Engine**
*STATS 507 Final Project*

## Project Overview
AI VibeMatch is an interactive Streamlit web application that utilizes Hugging Face Natural Language Processing (NLP) and K-Nearest Neighbors (KNN) machine learning to generate context-aware, 30-track Spotify playlists. 

Instead of traditional keyword searching, users input a natural language diary entry. The system utilizes a Zero-Shot Classification model (`facebook/bart-large-mnli`) to map textual sentiment to Russell's Circumplex Model of Affect, extracting audio features like Valence, Energy, Danceability, Acousticness, and Tempo.

Per the initial project proposal, this codebase strictly adheres to **Object-Oriented Programming (OOP)** principles via the `VibeMatchEngine` class, utilizes Python's built-in **Logging** for backend monitoring, and includes **Unit Tests** for robust execution.

## Repository Structure
* `app.py`: The main Streamlit application containing the UI, OOP classes, and ML pipelines.
* `data_preprocessing.ipynb`: The Jupyter Notebook demonstrating the data wrangling and Min-Max scaling applied to the original Spotify dataset.
* `processed_spotify_data.csv`: The cleaned dataset used by the application.
* `test_app.py`: Unit tests to verify data integrity and bounds.
* `requirements.txt`: Required library dependencies.
* `Final_Report.pdf`: The IEEE-formatted summary report outlining methodologies and results.

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone <YOUR-GITHUB-REPO-LINK-HERE>
cd <YOUR-REPO-FOLDER-NAME>
```
### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv .venv
source .venv/Scripts/activate  # On Windows
# source .venv/bin/activate    # On Mac/Linux
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the Web Application
```bash
streamlit run app.py
```
### 5. Run the Unit Tests
```bash
python -m unittest test_app.py
```
