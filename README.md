<img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/1dd685c9-90d6-43bd-89b5-24583967eb5d" /># 🎬 Movie Recommendation System (Hybrid Recommender)

A complete end-to-end movie recommendation system combining:

- Collaborative Filtering (ALS Matrix Factorization)
- Content-Based Filtering (TF-IDF + Cosine Similarity)
- Hybrid Recommendation Model
- User Profile-Based Recommendations

Built using Python, Pandas, Scikit-learn, and the Implicit library.

---

## 🚀 Features

### 🔹 Collaborative Filtering (ALS)
- Trained on 1M+ ratings
- Learns latent user and item embeddings
- Generates personalized recommendations

### 🔹 Content-Based Filtering
- Uses movie genres + tags
- TF-IDF vectorization
- Cosine similarity between movies

### 🔹 Hybrid Model
- Combines ALS + Content scores
- Weighted scoring (alpha blending)

### 🔹 User Profile Model
- Builds user preference vector from liked movies
- Recommends similar items via cosine similarity

---

## 🧠 System Architecture

1. Load raw data (ratings, movies, tags)
2. Preprocess and clean datasets
3. Build user-item matrix
4. Train ALS model
5. Build TF-IDF content matrix
6. Generate:
   - Collaborative recommendations
   - Content-based recommendations
   - Hybrid recommendations
   - User-profile recommendations
7. Evaluate model (optional module)

---

## 📊 Dataset

- MovieLens dataset (ratings, movies, tags)
- ~1,000,000 ratings used for training
- ~27,000 movies

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Implicit ALS
- TF-IDF Vectorization
- Cosine Similarity

---

## 📌 Recommendation Pipeline

1. Load MovieLens dataset
2. Clean and preprocess data
3. Create User-Item interaction matrix
4. Train ALS recommendation model
5. Build TF-IDF movie representation
6. Compute cosine similarity
7. Generate content-based recommendations
8. Combine collaborative and content-based predictions
9. Generate personalized recommendations

---

## ▶️ Example Output

```
Recommendations for User 175325

Forrest Gump (1994)
Batman Begins (2005)
Stand by Me (1986)
The Lord of the Rings: The Two Towers (2002)
The Prestige (2006)
...
```

---

## ⬇️ Installation

Clone the repository

```bash
git clone https://github.com/razbywork/Movie-Recommendation-System-Hybrid-Recommender-.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python main.py
```

---

## 🌐 Link to Dataset
https://www.kaggle.com/datasets/ayushimishra2809/movielens-dataset

---

## 👤 Author

Raz Ben-Yehuda
B.Sc. Industrial Engineering & Management
Specialization in Information Systems & Business Analytics
