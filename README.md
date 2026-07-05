# 🎬 Movie Recommendation System (Hybrid Recommender)

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
- Implicit (ALS)

---

## ▶️ How to Run

python main.py

---

## 📈 Example Output

Top-N movie recommendations per user
Similar movies based on content
Hybrid ranked results
User preference-based suggestions

---

## 📌 Project Structure

src/
 ├── data_loader.py
 ├── preprocessing.py
 ├── collaborative_model.py
 ├── content_model.py
 ├── hybrid_model.py

main.py

---

## 🌐 Link to Dataset
https://www.kaggle.com/datasets/ayushimishra2809/movielens-dataset

---

## 👤 Author

Raz Ben-Yehuda
B.Sc. Industrial Engineering & Management
Specialization in Information Systems & Business Analytics
