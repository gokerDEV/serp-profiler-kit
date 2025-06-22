import os
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
from typing import Dict, List, Tuple, Optional, Any
import logging
import argparse
from tqdm import tqdm
from trafilatura import extract

# Add the root directory to the Python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.lib.utils import Utils

# Download required NLTK data if not already downloaded
nltk.download('punkt', quiet=True)

class FeatureExtractor:
    """
    Extracts essential features from HTML files for content relevance analysis.
    
    Features extracted include:
    - query_in_title: If query appears in HTML title
    - query_in_h1: If query appears in first H1 heading
    - exact_query_in_title: If query appears in HTML title (exact match)
    - exact_query_in_h1: If query appears in first H1 heading (exact match)
    - query_density_body: Density of query in main text
    - semantic_similarity_title_query: Semantic similarity between title and query
    - semantic_similarity_content_query: Semantic similarity between content and query
    - word_count: Total word count of the main content
    """
    
    def __init__(
        self,
        input_file: str,
        raw_data_dir: str,
        output_file: str,
        model_name: str = 'all-mpnet-base-v2'
    ):
        """
        Initialize the feature extractor.
        
        Args:
            input_file: Path to the accepted.csv file
            raw_data_dir: Directory containing the raw HTML files
            output_file: Path to save the extracted features
            model_name: Name of the sentence-transformer model to use
        """
        self.input_file = input_file
        self.raw_data_dir = raw_data_dir
        self.output_file = output_file
        self.model_name = model_name
        
        # Set up logging
        self.logger = Utils.set_colorful_logging('FeatureExtractor')
        
        # Load data
        self.df = pd.read_csv(input_file)
        
        # Load the sentence transformer model
        self.logger.info(f"Loading {model_name} model...")
        self.model = SentenceTransformer(model_name)
        self.logger.info("Model loaded successfully")
        
        # Initialize results dataframe
        self.results = pd.DataFrame()
        
    def run(self):
        """Run the feature extraction process"""
        self.logger.info(f"Starting feature extraction for {len(self.df)} entries...")
        
        # Initialize results list
        results = []
        
        # Process each entry
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Extracting features"):
            try:
                # Extract features for the current entry
                features = self._extract_features(row)
                results.append(features)
            except Exception as e:
                self.logger.error(f"Error processing {row['file_name']}: {str(e)}")
                # Add a row with the URL and file_name but NaN for features
                features = {
                    "url": row["url"],
                    "file_name": row["file_name"],
                    "error": str(e)
                }
                results.append(features)
                
        # Create results DataFrame
        self.results = pd.DataFrame(results)
        
        # Save results
        self.results.to_csv(self.output_file, index=False)
        self.logger.info(f"Feature extraction completed. Results saved to {self.output_file}")
        
    def _extract_features(self, row: pd.Series) -> Dict[str, Any]:
        """
        Extract features from an HTML file
        
        Args:
            row: Row from the input DataFrame containing url, file_name, and query
            
        Returns:
            Dictionary of extracted features
        """
        url = row["url"]
        file_name = row["file_name"]
        
        # Preserve all original columns from the input file
        features = {col: row[col] for col in row.index}
        
        # Construct file paths
        html_path = os.path.join(self.raw_data_dir, "pages", f"{file_name}.html")
        
        # Check if the file exists
        if not os.path.exists(html_path):
            raise FileNotFoundError(f"HTML file not found: {html_path}")
        
        # Read HTML content
        with open(html_path, "r", encoding="utf-8", errors="replace") as f:
            html_content = f.read()
        
        # Parse HTML
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Get query directly from the dataframe row
        query = row.get("query", "")
        if not query:
            self.logger.warning(f"Query not found for {file_name}, using empty string")
        
        # Extract main content text using trafilatura
        main_text = extract(html_content) or ""
        
        # Extract HTML title from the page
        html_title = soup.title.text.strip() if soup.title else ""
        
        # 1. query_in_title: Check if query appears in HTML title
        # Check both exact match and all words appearing in title
        title_lower = html_title.lower()
        query_lower = query.lower()
        
        # Check for exact match
        exact_match_in_title = 1 if query_lower and query_lower in title_lower else 0
        
        # Check if all words from query appear in title
        if not exact_match_in_title and query:  # Check that query is not empty
            query_words = set(query_lower.split())
            if query_words:  # Check that query_words is not empty (handles whitespace-only queries)
                all_words_in_title = all(word in title_lower for word in query_words)
                query_in_title = 1 if all_words_in_title else 0
            else:  # If query exists but split results in empty set (e.g., query is just whitespace)
                query_in_title = 0
        else:
            query_in_title = exact_match_in_title
            
        features["query_in_title"] = query_in_title
        features["exact_query_in_title"] = exact_match_in_title
        
        # 2. query_in_h1: Check if query is in first H1 heading (only use the first H1)
        first_h1 = soup.find("h1")
        h1_text = first_h1.text.strip().lower() if first_h1 else ""
        
        # Check for exact match
        exact_match_in_h1 = 1 if query_lower and query_lower in h1_text else 0
        
        # Check if all words from query appear in H1
        if not exact_match_in_h1 and query and h1_text:  # Check that query and h1_text are not empty
            query_words = set(query_lower.split())
            if query_words:  # Check that query_words is not empty
                all_words_in_h1 = all(word in h1_text for word in query_words)
                query_in_h1 = 1 if all_words_in_h1 else 0
            else:
                query_in_h1 = 0
        else:
            query_in_h1 = exact_match_in_h1
            
        features["query_in_h1"] = query_in_h1
        features["exact_query_in_h1"] = exact_match_in_h1
        
        # 3. query_density_body: Calculate query density in main text
        features["query_density_body"] = self._calculate_query_density(main_text, query)
        
        # 4 & 5. Semantic Similarity features
        semantic_features = self._calculate_semantic_similarity(query, html_title, main_text)
        features.update(semantic_features)
        
        # 6. word_count: Count words in main text
        word_count = len([w for w in word_tokenize(main_text) if w.isalpha()]) if main_text else 0
        features["word_count"] = word_count
        
        return features
        
    def _calculate_query_density(self, main_text: str, query: str) -> float:
        """
        Calculate query density in main text
        
        Args:
            main_text: Main text content
            query: Query string
            
        Returns:
            Query density as a percentage
        """
        if not main_text or not query:
            return 0.0
            
        main_text_lower = main_text.lower()
        query_lower = query.lower()
        
        # Tokenize the main text
        tokens = word_tokenize(main_text_lower)
        
        # Keep only alpha tokens (words)
        tokens = [token for token in tokens if token.isalpha()]
        
        # Count total words
        total_words = len(tokens)
        
        if total_words == 0:
            return 0.0
            
        # Count query occurrence (full query)
        query_count = main_text_lower.count(query_lower)
        
        # Calculate query density
        query_density = (query_count / total_words) * 100
        
        return query_density
        
    def _calculate_semantic_similarity(
        self, 
        query: str, 
        html_title: str, 
        main_text: str
    ) -> Dict[str, float]:
        """
        Calculate semantic similarity between query and content
        
        Args:
            query: Query string
            html_title: HTML title text
            main_text: Main text content
            
        Returns:
            Dictionary with semantic similarity features
        """
        if not query:
            return {
                "semantic_similarity_title_query": 0.0,
                "semantic_similarity_content_query": 0.0
            }
            
        # Encode query
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Title similarity
        if html_title:
            title_embedding = self.model.encode(html_title, convert_to_tensor=True)
            title_similarity = float(util.pytorch_cos_sim(query_embedding, title_embedding).item())
        else:
            title_similarity = 0.0
            
        # Content similarity
        if main_text and len(main_text) > 50:  # Ensure there's enough content
            # For long texts, chunk it and compute average similarity
            if len(main_text) > 2000:
                chunks = self._chunk_text(main_text, max_length=1500, overlap=200)
                chunk_embeddings = self.model.encode(chunks, convert_to_tensor=True)
                similarities = util.pytorch_cos_sim(query_embedding, chunk_embeddings)
                content_similarity = float(similarities.mean().item())
            else:
                content_embedding = self.model.encode(main_text, convert_to_tensor=True)
                content_similarity = float(util.pytorch_cos_sim(query_embedding, content_embedding).item())
        else:
            content_similarity = 0.0
            
        return {
            "semantic_similarity_title_query": title_similarity,
            "semantic_similarity_content_query": content_similarity
        }
        
    def _chunk_text(self, text: str, max_length: int = 1500, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks for more effective semantic analysis
        
        Args:
            text: Text to chunk
            max_length: Maximum chunk length
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        tokens = word_tokenize(text)
        chunks = []
        
        if len(tokens) <= max_length:
            return [text]
            
        for i in range(0, len(tokens), max_length - overlap):
            chunk_tokens = tokens[i:i + max_length]
            chunks.append(" ".join(chunk_tokens))
            
            if i + max_length >= len(tokens):
                break
                
        return chunks

def main():
    """Main function to run the feature extractor"""
    parser = argparse.ArgumentParser(
        description='Extract essential features from HTML files for content relevance analysis'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/processed/accepted.csv',
        help='Input CSV file with URLs, file names, and queries'
    )
    parser.add_argument(
        '--raw_dir',
        type=str,
        default='data/raw',
        help='Directory containing raw HTML files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/features.csv',
        help='Output CSV file for extracted features'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='all-mpnet-base-v2',
        help='Name of the sentence-transformer model to use'
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize and run the feature extractor
    extractor = FeatureExtractor(
        input_file=args.input,
        raw_data_dir=args.raw_dir,
        output_file=args.output,
        model_name=args.model
    )
    
    extractor.run()
    
    print(f"✅ Feature extraction completed. Results saved to {args.output}")

if __name__ == "__main__":
    main() 