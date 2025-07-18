import streamlit as st
import numpy as np
import pandas as pd
import faiss
import requests
from sentence_transformers import SentenceTransformer
import ast
import os

# Set page config
st.set_page_config(
    page_title="Movie Recommender Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", "")  # You'll need to add this to secrets
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Cache the model loading
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Cache data loading
@st.cache_data
def load_embeddings_and_ids():
    """Load all embeddings and IDs from Data folder"""
    data_folder = "Data"
    
    # Load embeddings
    embeddings = np.load(os.path.join(data_folder, "embeddings.npy"))
    hybrid_embeddings = np.load(os.path.join(data_folder, "hybrid_embeddings.npy"))
    user_embeddings = np.load(os.path.join(data_folder, "user_embeddings.npy"))
    
    # Load IDs
    semantic_ids = np.load(os.path.join(data_folder, "semanticIds.npy"))
    hybrid_ids = np.load(os.path.join(data_folder, "hybridIds.npy"))
    
    return embeddings, hybrid_embeddings, user_embeddings, semantic_ids, hybrid_ids

@st.cache_resource
def build_faiss_indexes():
    """Build FAISS indexes for fast similarity search"""
    embeddings, hybrid_embeddings, user_embeddings, semantic_ids, hybrid_ids = load_embeddings_and_ids()
    
    # Normalize embeddings
    from sklearn.preprocessing import normalize
    embeddings = normalize(embeddings)
    hybrid_embeddings = normalize(hybrid_embeddings)
    
    # Build FAISS indexes
    semantic_index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product for normalized vectors
    semantic_index.add(embeddings.astype('float32'))
    
    hybrid_index = faiss.IndexFlatIP(hybrid_embeddings.shape[1])
    hybrid_index.add(hybrid_embeddings.astype('float32'))
    
    return semantic_index, hybrid_index, semantic_ids, hybrid_ids

def get_movie_details(movie_id):
    """Get movie details from TMDB API"""
    if not TMDB_API_KEY:
        return None
    
    try:
        url = f"{TMDB_BASE_URL}/movie/{movie_id}"
        params = {"api_key": TMDB_API_KEY}
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

def get_movie_poster_url(poster_path):
    """Get full poster URL from TMDB"""
    if poster_path:
        return f"{TMDB_IMAGE_BASE_URL}{poster_path}"
    return None

def search_movie_by_title(title):
    """Search for movies by title using TMDB API"""
    if not TMDB_API_KEY:
        return []
    
    try:
        url = f"{TMDB_BASE_URL}/search/movie"
        params = {
            "api_key": TMDB_API_KEY,
            "query": title,
            "include_adult": False
        }
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            results = response.json().get('results', [])
            return results[:10]  # Return top 10 results
        else:
            return []
    except:
        return []

def search_movies_semantic(query, top_k=10):
    """Search movies using semantic embeddings"""
    model = load_sentence_transformer()
    semantic_index, hybrid_index, semantic_ids, hybrid_ids = build_faiss_indexes()
    
    # Encode query
    query_embedding = model.encode([query])
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    # Search
    scores, indices = semantic_index.search(query_embedding.astype('float32'), top_k)
    
    # Get movie IDs
    movie_ids = [semantic_ids[i] for i in indices[0]]
    scores = scores[0]
    
    return movie_ids, scores

def get_similar_movies_hybrid(movie_id, top_k=10):
    """Get similar movies using hybrid embeddings"""
    semantic_index, hybrid_index, semantic_ids, hybrid_ids = build_faiss_indexes()
    
    # Find movie in hybrid IDs
    try:
        movie_index = np.where(hybrid_ids == int(movie_id))[0][0]
    except:
        return [], []
    
    # Get embedding for this movie
    embeddings, hybrid_embeddings, user_embeddings, semantic_ids_data, hybrid_ids_data = load_embeddings_and_ids()
    movie_embedding = hybrid_embeddings[movie_index].reshape(1, -1)
    movie_embedding = movie_embedding / np.linalg.norm(movie_embedding)
    
    # Search for similar movies
    scores, indices = hybrid_index.search(movie_embedding.astype('float32'), top_k + 1)  # +1 to exclude self
    
    # Remove self from results
    filtered_indices = [i for i in indices[0] if i != movie_index][:top_k]
    filtered_scores = [scores[0][i] for i, idx in enumerate(indices[0]) if idx != movie_index][:top_k]
    
    # Get movie IDs
    movie_ids = [hybrid_ids[i] for i in filtered_indices]
    
    return movie_ids, filtered_scores

def get_similar_movies_semantic(movie_id, top_k=10):
    """Get similar movies using semantic embeddings"""
    semantic_index, hybrid_index, semantic_ids, hybrid_ids = build_faiss_indexes()
    
    # Find movie in semantic IDs
    try:
        movie_index = np.where(semantic_ids == int(movie_id))[0][0]
    except:
        return [], []
    
    # Get embedding for this movie
    embeddings, hybrid_embeddings, user_embeddings, semantic_ids_data, hybrid_ids_data = load_embeddings_and_ids()
    movie_embedding = embeddings[movie_index].reshape(1, -1)
    movie_embedding = movie_embedding / np.linalg.norm(movie_embedding)
    
    # Search for similar movies
    scores, indices = semantic_index.search(movie_embedding.astype('float32'), top_k + 1)  # +1 to exclude self
    
    # Remove self from results
    filtered_indices = [i for i in indices[0] if i != movie_index][:top_k]
    filtered_scores = [scores[0][i] for i, idx in enumerate(indices[0]) if idx != movie_index][:top_k]
    
    # Get movie IDs
    movie_ids = [semantic_ids[i] for i in filtered_indices]
    
    return movie_ids, filtered_scores

def display_movie_card(movie_id, score=None, show_recommend_button=True, card_index=None):
    """Display a movie card with poster, title, and details"""
    movie_details = get_movie_details(movie_id)
    
    if movie_details:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            poster_url = get_movie_poster_url(movie_details.get('poster_path'))
            if poster_url:
                st.image(poster_url, width=150)
            else:
                st.write("No poster available")
        
        with col2:
            st.subheader(movie_details.get('title', 'Unknown Title'))
            st.write(f"**Release Date:** {movie_details.get('release_date', 'Unknown')}")
            st.write(f"**Rating:** {movie_details.get('vote_average', 'N/A')}/10")
            st.write(f"**Overview:** {movie_details.get('overview', 'No overview available')[:200]}...")
            
            if score is not None:
                st.write(f"**Similarity Score:** {score:.3f}")
            
            if show_recommend_button:
                # Use a unique key for each button with card index
                button_key = f"rec_{movie_id}_{card_index}"
                if st.button(f"Get Recommendations", key=button_key):
                    # Clear any existing recommendations first
                    st.session_state.show_quick_recommendations = False
                    st.session_state.quick_rec_movie_id = None
                    # Set the new movie and trigger rerun
                    st.session_state.quick_rec_movie_id = movie_id
                    st.session_state.show_quick_recommendations = True
                    st.rerun()
    else:
        # Fallback display when TMDB API is not available
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.write("🎬")  # Movie emoji as placeholder
        
        with col2:
            st.subheader(f"Movie ID: {movie_id}")
            st.write("**Details:** Not available (TMDB API key required)")
            if score is not None:
                st.write(f"**Similarity Score:** {score:.3f}")
            
            if show_recommend_button:
                # Use a unique key for each button with card index
                button_key = f"rec_fallback_{movie_id}_{card_index}"
                if st.button(f"Get Recommendations", key=button_key):
                    # Clear any existing recommendations first
                    st.session_state.show_quick_recommendations = False
                    st.session_state.quick_rec_movie_id = None
                    # Set the new movie and trigger rerun
                    st.session_state.quick_rec_movie_id = movie_id
                    st.session_state.show_quick_recommendations = True
                    st.rerun()

def main():
    st.title("🎬 Movie Recommender Engine")
    st.markdown("Discover movies using semantic search and get personalized recommendations!")
    
    # Initialize session state
    if 'show_recommendations' not in st.session_state:
        st.session_state.show_recommendations = False
    if 'selected_movie_id' not in st.session_state:
        st.session_state.selected_movie_id = None
    if 'show_quick_recommendations' not in st.session_state:
        st.session_state.show_quick_recommendations = False
    if 'quick_rec_movie_id' not in st.session_state:
        st.session_state.quick_rec_movie_id = None
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    
    # Check if TMDB API key is available
    if not TMDB_API_KEY:
        st.warning("⚠️ TMDB API key not found. Please add your TMDB API key to the secrets to see movie posters and details.")
        st.markdown("You can get a free API key from [TMDB](https://www.themoviedb.org/settings/api)")
    
    # Sidebar for options
    st.sidebar.header("Options")
    search_type = st.sidebar.selectbox(
        "Search Type",
        ["Semantic Search", "Movie Recommendations"]
    )
    
    if search_type == "Semantic Search":
        st.header("🔍 Semantic Search")
        
        # Search input
        query = st.text_input("Enter a movie description, genre, or any keywords:", 
                            placeholder="e.g., 'action movie with robots and time travel'")
        
        # Number of results
        top_k = st.slider("Number of results:", 1, 20, 10)
        
        if st.button("Search Movies") and query:
            with st.spinner("Searching for movies..."):
                movie_ids, scores = search_movies_semantic(query, top_k)
            
            # Store results in session state
            st.session_state.search_results = list(zip(movie_ids, scores))
            st.session_state.search_query = query
        
        # Display search results from session state
        if st.session_state.search_results:
            st.subheader(f"Top {len(st.session_state.search_results)} Results for: '{st.session_state.search_query}'")
            
            # Add a clear search button
            if st.button("Clear Search Results"):
                st.session_state.search_results = []
                st.session_state.search_query = ""
                st.session_state.show_quick_recommendations = False
                st.session_state.quick_rec_movie_id = None
                st.rerun()
            
            for i, (movie_id, score) in enumerate(st.session_state.search_results):
                st.markdown(f"### {i+1}. Movie Result")
                display_movie_card(movie_id, score, card_index=i)
                st.markdown("---")
    
    elif search_type == "Movie Recommendations":
        st.header("🎯 Movie Recommendations")
        
        # Search method selection
        search_method = st.radio(
            "Search by:",
            ["Movie Title", "Movie ID"],
            horizontal=True
        )
        
        selected_movie_id = None
        
        if search_method == "Movie Title":
            # Movie title input
            movie_title = st.text_input("Enter a movie title:", 
                                      placeholder="e.g., 'Fight Club', 'The Matrix', 'Inception'")
            
            if movie_title:
                if not TMDB_API_KEY:
                    st.warning("⚠️ Movie title search requires a TMDB API key. Please add your API key to use this feature, or use Movie ID search instead.")
                    st.info("You can still use Movie ID search below. Some popular movie IDs: Fight Club (550), The Matrix (603), Inception (27205)")
                else:
                    # Search for movies by title
                    search_results = search_movie_by_title(movie_title)
                    
                    if search_results:
                        st.subheader(f"Search Results for '{movie_title}':")
                        
                        # Create a selectbox with movie options
                        movie_options = []
                        movie_id_map = {}
                        
                        for movie in search_results:
                            title = movie.get('title', 'Unknown Title')
                            release_date = movie.get('release_date', 'Unknown')
                            year = release_date.split('-')[0] if release_date and release_date != 'Unknown' else 'Unknown'
                            movie_id = movie.get('id')
                            
                            display_text = f"{title} ({year})"
                            movie_options.append(display_text)
                            movie_id_map[display_text] = movie_id
                        
                        selected_movie_text = st.selectbox(
                            "Select a movie:",
                            movie_options,
                            key="movie_selector"
                        )
                        
                        if selected_movie_text:
                            selected_movie_id = movie_id_map[selected_movie_text]
                            
                            # Show selected movie details
                            st.subheader("Selected Movie:")
                            display_movie_card(selected_movie_id, show_recommend_button=False, card_index=0)
                            
                    elif movie_title:
                        st.warning(f"No movies found for '{movie_title}'. Please try a different title.")
        
        else:  # Movie ID search
            # Movie ID input
            movie_id_input = st.text_input("Enter a TMDB Movie ID:", 
                                         placeholder="e.g., 550 (Fight Club)")
            
            if movie_id_input:
                try:
                    selected_movie_id = int(movie_id_input)
                    
                    # Show the input movie first
                    st.subheader("Selected Movie:")
                    display_movie_card(selected_movie_id, show_recommend_button=False, card_index=0)
                    
                except ValueError:
                    st.error("Please enter a valid movie ID (number).")
        
        # Recommendation options (shown only when a movie is selected)
        if selected_movie_id:
            st.markdown("---")
            
            # Recommendation type
            rec_type = st.selectbox(
                "Recommendation Type",
                ["Hybrid (Content + Collaborative)", "Semantic (Content-based only)"]
            )
            
            # Number of recommendations
            top_k = st.slider("Number of recommendations:", 1, 20, 10)
            
            if st.button("Get Recommendations"):
                with st.spinner("Finding similar movies..."):
                    if rec_type == "Hybrid (Content + Collaborative)":
                        movie_ids, scores = get_similar_movies_hybrid(selected_movie_id, top_k)
                    else:
                        movie_ids, scores = get_similar_movies_semantic(selected_movie_id, top_k)
                
                if movie_ids:
                    st.subheader(f"Top {len(movie_ids)} Recommended Movies:")
                    
                    for i, (rec_movie_id, score) in enumerate(zip(movie_ids, scores)):
                        st.markdown(f"### {i+1}. Recommendation")
                        display_movie_card(rec_movie_id, score, card_index=f"rec_{i}")
                        st.markdown("---")
                else:
                    st.warning("No recommendations found for this movie ID.")
    
    # Handle quick recommendation clicks from movie cards
    if st.session_state.show_quick_recommendations and st.session_state.quick_rec_movie_id:
        st.markdown("---")
        st.subheader("🎯 Quick Recommendations:")
        st.info(f"Generating recommendations for Movie ID: {st.session_state.quick_rec_movie_id}")
        
        # Show the selected movie
        st.markdown("**Based on:**")
        display_movie_card(st.session_state.quick_rec_movie_id, show_recommend_button=False, card_index="base")
        
        # Use a unique key for spinner to force refresh when movie changes
        spinner_key = f"spinner_{st.session_state.quick_rec_movie_id}"
        with st.spinner("Finding similar movies..."):
            # Use hybrid recommendations by default, fallback to semantic
            movie_ids, scores = get_similar_movies_hybrid(st.session_state.quick_rec_movie_id, 10)
            
            if not movie_ids:  # Fallback to semantic if hybrid doesn't work
                movie_ids, scores = get_similar_movies_semantic(st.session_state.quick_rec_movie_id, 10)
        
        if movie_ids:
            st.markdown("**Recommended Movies:**")
            for i, (rec_movie_id, score) in enumerate(zip(movie_ids, scores)):
                st.markdown(f"### {i+1}. Recommendation")
                # Use unique card index that includes the source movie ID
                card_index = f"quick_{st.session_state.quick_rec_movie_id}_{i}"
                display_movie_card(rec_movie_id, score, card_index=card_index)
                st.markdown("---")
        else:
            st.warning("No recommendations found for this movie.")
        
        # Reset the quick recommendation state
        if st.button("Clear Quick Recommendations"):
            st.session_state.show_quick_recommendations = False
            st.session_state.quick_rec_movie_id = None
            st.rerun()
    
    # Handle recommendation clicks from search results (legacy)
    if st.session_state.show_recommendations and st.session_state.selected_movie_id:
        st.markdown("---")
        st.subheader("🎯 Recommendations based on your selection:")
        
        with st.spinner("Finding similar movies..."):
            # Use hybrid recommendations by default
            movie_ids, scores = get_similar_movies_hybrid(st.session_state.selected_movie_id, 10)
            
            if not movie_ids:  # Fallback to semantic if hybrid doesn't work
                movie_ids, scores = get_similar_movies_semantic(st.session_state.selected_movie_id, 10)
        
        if movie_ids:
            for i, (rec_movie_id, score) in enumerate(zip(movie_ids, scores)):
                st.markdown(f"### {i+1}. Recommendation")
                display_movie_card(rec_movie_id, score, card_index=f"legacy_{i}")
                st.markdown("---")
        else:
            st.warning("No recommendations found for this movie.")
        
        # Reset the recommendation state
        if st.button("Clear Recommendations"):
            st.session_state.show_recommendations = False
            st.session_state.selected_movie_id = None
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit, FAISS, and Sentence Transformers")

if __name__ == "__main__":
    main()
