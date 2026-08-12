## 🛍️ Hybrid Amazon Product Recommendation System

A content-based **product recommendation engine** that suggests similar Amazon products by blending **Natural Language Processing (NLP)** with classic **Machine Learning**. Built end-to-end — from raw data to a deployed Streamlit app.

---

## 📌 About the Project

This project recommends Amazon products that are most similar to a selected item by analyzing:

- Product name
- Main category & sub-category
- Ratings & number of ratings
- Price & discount percentage

To make the recommendations meaningful, **Sentence Transformer embeddings** capture the semantic meaning of product names/descriptions, while numerical attributes (price, ratings, discount, etc.) are scaled and combined with those embeddings to form a **hybrid feature vector**. Similar products are then retrieved using **K-Nearest Neighbors (KNN)** with **Cosine Similarity**.

The project follows a full ML workflow: **Data Cleaning → EDA → Feature Engineering → Model Training → Streamlit Deployment**.

---

## 🚀 Key Features

-  Search and recommend similar Amazon products
-  Hybrid recommendation combining NLP + numerical features
-  Semantic understanding via Sentence Transformer embeddings
-  Fast, similarity-based recommendations using KNN + Cosine Similarity
-  Product images and direct Amazon links shown in results
-  Clean, interactive Streamlit web interface

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| ML / Similarity | Scikit-learn, Joblib |
| NLP | Sentence Transformers, Transformers, Torch |
| Visualization | Matplotlib, Seaborn, Plotly |
| Web App | Streamlit |
| Image Handling | Pillow |

---

## 📁 Project Structure

```
xopuntech_Internship/
├── Datasets/                      # Raw / processed data (kept out of Git via .gitignore)
├── notebooks/
│   ├── Data_cleaning.ipynb        # Cleaning & preprocessing raw product data
│   ├── EDA.ipynb                  # Exploratory Data Analysis
│   ├── Feature_Engineering.ipynb  # Embeddings + scaled numerical features
│   └── Model.ipynb                # KNN model training & evaluation
├── Stremlit/
│   ├── .streamlit/                # Streamlit config
│   ├── assets/                    # Static assets used by the app
│   ├── static/                    # Additional static files
│   └── App.py                     # Streamlit application entry point
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MrinmoyTalukdar27/xopuntech_Internship.git
   cd xopuntech_Internship
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

Run the Streamlit app locally:

```bash
cd Stremlit
streamlit run App.py
```

This will launch the app in your browser, where you can search for a product and view a ranked list of similar recommendations along with images and Amazon links.

If you want to explore or reproduce the pipeline itself, run the notebooks in this order:-

1. `notebooks/Data_cleaning.ipynb`
2. `notebooks/EDA.ipynb`
3. `notebooks/Feature_Engineering.ipynb`
4. `notebooks/Model.ipynb`

---

## 🧠 How It Works

1. **Data Cleaning** – Handle missing values, fix data types, and standardize product attributes.
2. **EDA** – Explore category distribution, pricing patterns, and rating trends.
3. **Feature Engineering** – Generate Sentence Transformer embeddings for product text and scale numerical features (price, ratings, discount %).
4. **Hybrid Feature Vector** – Concatenate text embeddings with scaled numerical features.
5. **Model Training** – Fit a KNN model using Cosine Similarity on the hybrid vectors.
6. **Deployment** – Serve recommendations through an interactive Streamlit interface.

---

## 🎯 Project Objective

The goal of this project is to build an end-to-end product recommendation system that suggests relevant products based on semantic and numerical similarity — demonstrating the full ML pipeline from data preprocessing to deployment. It serves as a practical portfolio project for Data Science / Machine Learning roles, built as part of an internship at **Xopuntech**.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.

## 📄 License

This project currently has no license specified. Consider adding one (e.g., MIT) if you'd like others to freely reuse the code.

## 👤 Author

**Mrinmoy Talukdar**
GitHub: [@MrinmoyTalukdar27](https://github.com/MrinmoyTalukdar27)
LinkedIn: [mrinmoy-talukdar-5867ab3b9](https://www.linkedin.com/in/mrinmoy-talukdar-5867ab3b9)
