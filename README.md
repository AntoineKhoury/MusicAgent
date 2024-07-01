# 🎵 README.md for Music Production and Discussion Project 🎶

## 🌟 Introduction
Welcome to the 🚀 Music Analysis and Discussion Project! This innovative project is crafted for music enthusiasts and creators who wish to dive deep into the world of musical similarities and engaging discussions with AI. Leveraging a blend of ElasticSearch, cutting-edge machine learning models, and sophisticated audio processing techniques, we unlock new insights into your composed tracks. Whether you're curious about the nuances of your music or eager to connect with like-minded individuals, this project is your gateway to a new auditory experience.

## 🛠 Components
Dive into the core of our project with these two essential scripts:
1. `musicAgent.py` - Your go-to tool for analyzing songs and uncovering similar tracks.
2. `creatingElasticDatabase.py` - The backbone script that processes and injects a music dataset into the heart of an ElasticSearch database.

## 📋 Setup and Requirements

### 📚 Dependencies
Before embarking on your musical journey, ensure these dependencies are on board:
- Python 3.8.0 🐍
- pandas 🐼 (1.2.4)
- numpy 🔢 (1.22.4)
- elasticsearch 🔍 (7.13.0)
- librosa 🎵 (0.10.1)
- scipy 🧪 (1.6.3)
- transformers 🤖 (4.17.0)
- torch 🔥 (1.9.0)

### 🔐 ElasticSearch Setup
For the `creatingElasticDatabase.py` adventure:
- Make sure ElasticSearch is actively running on your local machine.
- Tweaks for `elasticsearch.yml`:
  - `xpack.security.enabled: false`
  - `xpack.security.http.ssl.enabled: false`
  
🚨 Remember, these settings are tailored for local fun! For broader horizons, secure your ElasticSearch.

## 🎤 Usage

### 🎧 Music Agent
Unleash the power of `musicAgent.py` to analyze a song and discover similar melodies:
```bash
python3 musicAgent.py "<Song Name>.wav"
```
👉 Don't forget: Run `creatingElasticDatabase.py` and keep ElasticSearch up and running.

### 📚 Creating Elastic Database
Embark on the `creatingElasticDatabase.py` mission to populate ElasticSearch with musical data:
```bash
python3 creatingElasticDatabase.py
```

## 🚧 Limitations and Future Improvements
Our current journey faces computational challenges, affecting the vibrancy of discussions. But fear not, the future holds:
- ☁️ Cloud-based model computations for soaring performance.
- 🛠 Code optimizations for smoother local symphonies.

## 🛡 Privacy and Data Handling
Your music's privacy is our top chart! We've engineered everything to run locally, keeping your tunes safe and sound.

## 🤝 Contributing
Got a melody of ideas or noticed a discordant note? Your contributions are music to our ears! Feel free to join our band and help us improvise.

## 📜 License
A tip of the hat to FMA and Hugging Face's "af1tang/personaGPT" for their code snippets. While we harmonize with their tunes, we don't own their compositions. We don't pretend to ahve any rights over their code or content.

## 📬 Contact
Antoine Khoury 🎸
[antoine.r.a.khoury@gmail.com](mailto:antoine.r.a.khoury@gmail.com)

---

🌈 Let's make music production more colorful and engaging together! 🎉