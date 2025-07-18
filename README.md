# 🎬 Movie Recommender Engine

A sophisticated movie recommendation system that combines machine learning, semantic search, and collaborative filtering to provide personalized movie recommendations. Built with Streamlit for an interactive web interface and powered by FAISS for lightning-fast similarity searches.

## ✨ Features

### 🔍 **Semantic Search**
- Search movies using natural language queries
- Find movies by describing plot, genre, mood, or themes
- Example: "action movie with robots and time travel"

### 🎯 **Smart Recommendations**
- **Hybrid Recommendations**: Combines content-based and collaborative filtering
- **Content-based**: Analyzes movie plots, genres, and metadata
- **Collaborative**: Uses user ratings and viewing patterns
- **Semantic**: Deep understanding of movie themes and content

### 🚀 **Advanced Technology**
- **FAISS Integration**: Lightning-fast similarity search with Facebook's FAISS library
- **Sentence Transformers**: State-of-the-art semantic embeddings using all-MiniLM-L6-v2
- **Pre-computed Embeddings**: Instant recommendations with no processing delay
- **TMDB API**: Real-time movie posters, ratings, and detailed information

### 💻 **Interactive Interface**
- Clean, intuitive Streamlit web interface
- Search by movie title or ID
- One-click recommendations from any movie card
- Visual movie cards with posters and details
- Responsive design for all screen sizes

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/lakshaykun/MoviesRecommenderEngine.git
cd MoviesRecommenderEngine
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Setup TMDB API Key
1. Get a free API key from [The Movie Database (TMDB)](https://www.themoviedb.org/settings/api)
2. Create the secrets directory and file:
   ```bash
   mkdir .streamlit
   ```
3. Create `.streamlit/secrets.toml` and add your API key:
   ```toml
   TMDB_API_KEY = "your_api_key_here"
   ```

### Step 5: Run the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🎮 How to Use

### 1. Semantic Search
- Navigate to the **Semantic Search** tab
- Enter descriptive queries like:
  - "romantic comedy with happy ending"
  - "dark thriller with plot twists"
  - "sci-fi movies about artificial intelligence"
- Adjust the number of results (1-20)
- Click **Search Movies** to find matching films

### 2. Movie Recommendations
- Go to the **Movie Recommendations** tab
- Choose your search method:

#### **Search by Title**
- Enter a movie title (e.g., "Fight Club", "The Matrix")
- Select the correct movie from the dropdown
- Choose recommendation type and number of results
- Click **Get Recommendations**

#### **Search by ID**
- Enter a TMDB movie ID (e.g., 550 for Fight Club)
- View the movie details
- Get recommendations using hybrid or semantic algorithms

### 3. Quick Recommendations
- Click the **Get Recommendations** button on any movie card
- Instantly view similar movies based on that selection
- Each recommendation includes similarity scores

## 📊 Project Architecture

### Data Pipeline
```
Raw Movie Data → Preprocessing → Feature Engineering → Embeddings Generation → FAISS Indexing
```

### Recommendation Types
1. **Semantic**: Content-based using movie plots and descriptions
2. **Hybrid**: Combines content and collaborative filtering
3. **User-based**: Collaborative filtering using user ratings

### Technology Stack
- **Frontend**: Streamlit (Python web framework)
- **ML/AI**: Sentence Transformers, FAISS, scikit-learn
- **APIs**: TMDB API for movie data
- **Data**: NumPy, Pandas for data processing

## 📁 Project Structure

```
MoviesRecommenderEngine/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── MovieRecommenderEngine.ipynb    # Jupyter notebook with data processing
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore file
├── .streamlit/
│   └── secrets.toml               # TMDB API key (create this file)
└── Data/
    ├── embeddings.npy             # Semantic embeddings
    ├── hybrid_embeddings.npy      # Hybrid embeddings
    ├── user_embeddings.npy        # User-based embeddings
    ├── semanticIds.npy            # Movie IDs for semantic search
    └── hybridIds.npy              # Movie IDs for hybrid recommendations
```

## 🚀 Quick Start

```bash
# Navigate to project directory
cd MoviesRecommenderEngine

# Activate virtual environment (if using one)
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Run the application
streamlit run app.py
```

## 📈 Performance & Scalability

### Optimization Features
- **Caching**: Streamlit caching for models and data loading
- **Pre-computed Embeddings**: No real-time computation needed
- **FAISS Indexing**: Optimized for large-scale similarity search
- **Efficient Memory Usage**: Normalized embeddings for reduced memory footprint

### Dataset Information
- **Movies**: 45,000+ movies from "The Movies Dataset" (Kaggle)
- **Ratings**: User ratings for collaborative filtering
- **Metadata**: Genres, keywords, cast, crew information
- **Embeddings**: Pre-computed vectors for instant recommendations

## 🔍 Example Queries

### Semantic Search Examples
```
"romantic movies set in paris"
"action movies with superheroes"
"psychological thrillers with unreliable narrators"
"animated movies for kids"
"sci-fi movies about time travel"
"horror movies with supernatural elements"
"comedies about friendship"
"war movies based on true stories"
```

### Popular Movie IDs for Testing
- Fight Club: `550`
- The Matrix: `603`
- Inception: `27205`
- The Dark Knight: `155`
- Pulp Fiction: `680`
- Forrest Gump: `13`
- The Shawshank Redemption: `278`
- The Godfather: `238`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **The Movies Dataset**: Kaggle dataset by Rounak Banik
- **TMDB**: The Movie Database for movie information and posters
- **Streamlit**: For the amazing web framework
- **Facebook AI**: For the FAISS library
- **Sentence Transformers**: For semantic embedding models

## 📧 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/lakshaykun/MoviesRecommenderEngine/issues) page
2. Create a new issue with detailed description
3. Include system information and error messages

## 🔮 Future Enhancements

- [ ] Real-time model training
- [ ] User preference learning
- [ ] Multi-language support
- [ ] Advanced filtering options
- [ ] Watchlist functionality
- [ ] Social features and sharing
- [ ] Mobile app version
- [ ] API endpoints for integration

---

**Built with ❤️ using Python, Streamlit, FAISS, and Sentence Transformers**

*Last updated: July 2025*
    ├── embeddings.npy       # Semantic embeddings
    ├── hybrid_embeddings.npy # Hybrid embeddings
    ├── user_embeddings.npy  # User-based embeddings
    ├── semanticIds.npy      # Movie IDs for semantic search
    └── hybridIds.npy        # Movie IDs for hybrid recommendations
```

## Technical Details

- **Semantic Search**: Uses Sentence Transformers (`all-MiniLM-L6-v2`) for encoding queries and movie content
- **Hybrid Recommendations**: Combines content-based (60%) and collaborative filtering (40%) embeddings
- **FAISS Indexing**: Uses IndexFlatIP for fast cosine similarity search
- **Embeddings**: Pre-computed embeddings stored as NumPy arrays for efficient loading

## Example Usage

1. **Semantic Search**: Search for "action movie with robots and time travel"
2. **Movie Recommendations**: Enter movie ID 550 (Fight Club) to get similar movies
3. **Interactive Mode**: Click "Get Recommendations" on any movie card for instant suggestionsg a project that has a large scale of movie dataset. It will have the following features – 
1.	Using the “The Movies Dataset” - https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset.
2.	Cleaning the dataset.
3.	Generates embedding using movie overview and other metadata.
4.	Uses ratings from users to generate user based movie embeddings.
5.	Combines both the methods optimally. To get the best results.
6.	Uses FAISS for fast execution of the recommender pipeline.
7.	Uses this recommender system in a simple Streamlit based GUI.
8.	GUI should have semantic search query feature. Also clicking on a movie should recommend us similar movies to it.
Help me with this project.
