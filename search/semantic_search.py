"""
Semantic Image Search Module

Provides semantic image search using MiniLM embeddings.
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
from PIL import Image

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticImageSearch:
    """
    Semantic image search using sentence-transformers (MiniLM).

    Encodes image captions and queries for semantic similarity matching.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embeddings_file: Optional[str] = None,
    ):
        """
        Initialize semantic image search.

        Args:
            model_name: Name of sentence-transformer model
            embeddings_file: Path to store embeddings JSON
        """
        self.model_name = model_name
        self.embeddings_file = embeddings_file or "data/image_embeddings.json"
        self.model = None
        self.image_index: Dict[str, Dict] = {}
        self._load_embeddings()
        self._init_model()

    def _init_model(self):
        """Initialize sentence-transformer model."""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading sentence-transformer model: {self.model_name}")
            device = (
                "mps"
                if hasattr(__import__("torch").backends, "mps")
                and __import__("torch").backends.mps.is_available()
                else "cpu"
            )
            self.model = SentenceTransformer(self.model_name, device=device)
            logger.info(f"Model loaded on {device}")
        except Exception as e:
            logger.warning(f"Failed to load sentence-transformer: {e}")
            logger.info("Using fallback embedding method")
            self.model = None

    def _load_embeddings(self):
        """Load existing embeddings from file."""
        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, "r") as f:
                    self.image_index = json.load(f)
                logger.info(f"Loaded {len(self.image_index)} image embeddings")
            except Exception as e:
                logger.warning(f"Failed to load embeddings: {e}")
                self.image_index = {}

    def _save_embeddings(self):
        """Save embeddings to file."""
        os.makedirs(os.path.dirname(self.embeddings_file) or "data", exist_ok=True)
        try:
            with open(self.embeddings_file, "w") as f:
                json.dump(self.image_index, f, indent=2)
            logger.info(f"Saved {len(self.image_index)} image embeddings")
        except Exception as e:
            logger.warning(f"Failed to save embeddings: {e}")

    def _get_embedding_fallback(self, text: str) -> np.ndarray:
        """Fallback embedding using simple hash + random for MVP."""
        # Simple hash-based embedding for fallback
        hash_val = hash(text) % 10000
        np.random.seed(hash_val)
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to embedding vector.

        Args:
            text: Text to encode

        Returns:
            Embedding vector (384-dim)
        """
        if self.model is None:
            return self._get_embedding_fallback(text)

        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)

    def add_image(self, image_path: str, caption: str) -> str:
        """
        Add image with caption to the index.

        Args:
            image_path: Path to image file
            caption: Image description/caption

        Returns:
            Image ID
        """
        # Generate image ID
        image_id = f"img_{len(self.image_index)}_{os.path.basename(image_path)}"

        # Get embedding
        embedding = self.encode_text(caption)

        # Store in index
        self.image_index[image_id] = {
            "path": image_path,
            "caption": caption,
            "embedding": embedding.tolist(),
        }

        self._save_embeddings()
        logger.info(f"Added image {image_id}: {caption[:50]}...")

        return image_id

    def search(self, query: str, top_k: int = 5, threshold: float = 0.3) -> List[Dict]:
        """
        Search images by semantic similarity.

        Args:
            query: Natural language query
            top_k: Number of results to return
            threshold: Minimum similarity score

        Returns:
            List of matching images with scores
        """
        if not self.image_index:
            return []

        # Encode query
        query_embedding = self.encode_text(query)

        # Compare with all images
        results = []
        for image_id, data in self.image_index.items():
            embedding = np.array(data["embedding"])

            # Cosine similarity
            similarity = np.dot(query_embedding, embedding)

            if similarity >= threshold:
                results.append(
                    {
                        "image_id": image_id,
                        "path": data["path"],
                        "caption": data["caption"],
                        "score": float(similarity),
                    }
                )

        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    def search_video_events(
        self, query: str, events: List[Dict], top_k: int = 5
    ) -> List[Dict]:
        """
        Search video events by semantic similarity.

        Args:
            query: Natural language query
            events: List of video events with descriptions
            top_k: Number of results

        Returns:
            List of matching events
        """
        if not events:
            return []

        # Encode query
        query_embedding = self.encode_text(query)

        # Compare with each event
        results = []
        for event in events:
            # Use description or generate from surprise score
            text = event.get("description", "")

            if text:
                event_embedding = self.encode_text(text)
                similarity = np.dot(query_embedding, event_embedding)

                results.append(
                    {
                        "frame_index": event.get("frame_index", 0),
                        "description": text,
                        "surprise_score": event.get("surprise_score", 0),
                        "score": float(similarity),
                    }
                )

        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    def get_all_images(self) -> List[Dict]:
        """Get all indexed images."""
        return [{"image_id": k, **v} for k, v in self.image_index.items()]

    def delete_image(self, image_id: str) -> bool:
        """Delete image from index."""
        if image_id in self.image_index:
            del self.image_index[image_id]
            self._save_embeddings()
            return True
        return False

    def clear_index(self):
        """Clear all embeddings."""
        self.image_index.clear()
        self._save_embeddings()
        logger.info("Cleared image index")


class ImageSearchAPI:
    """
    Flask API wrapper for image search functionality.
    """

    def __init__(self):
        self.search = SemanticImageSearch()

    def upload_image(self, file_path: str, caption: str = None) -> Dict:
        """
        Handle image upload.

        Args:
            file_path: Path to uploaded image
            caption: Optional caption (will generate if not provided)

        Returns:
            Result dict with image_id and caption
        """
        # Validate image
        try:
            img = Image.open(file_path)
            img.verify()
        except Exception as e:
            return {"error": f"Invalid image: {e}"}

        # Generate caption if not provided
        if caption is None:
            caption = self._generate_caption(file_path)

        # Add to index
        image_id = self.search.add_image(file_path, caption)

        return {
            "image_id": image_id,
            "caption": caption,
            "path": file_path,
        }

    def _generate_caption(self, image_path: str) -> str:
        """Generate caption for image using SmolVLM."""
        # Try to use the encoder from COG-JEPA
        try:
            from encoder.smolvlm_encoder import SmolVLMEncoder

            encoder = SmolVLMEncoder()
            img = Image.open(image_path).resize((224, 224))
            latent = encoder.encode(img)

            # Use simple heuristic for caption
            latent_mean = np.mean(latent)
            latent_std = np.std(latent)

            if latent_std > 0.5:
                return "Complex scene with multiple elements and high variation"
            elif latent_mean > 0:
                return "Image with notable features and patterns"
            else:
                return "Image with subtle details and low contrast"
        except Exception as e:
            logger.warning(f"Failed to generate caption: {e}")
            return f"Image from {os.path.basename(image_path)}"

    def search_images(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search images by query."""
        return self.search.search(query, top_k=top_k)

    def search_all(
        self, query: str, images: List[Dict] = None, events: List[Dict] = None
    ) -> Dict:
        """
        Cross-modal search across images and video events.

        Args:
            query: Natural language query
            images: Optional list of images to search
            events: Optional list of video events to search

        Returns:
            Dict with 'images' and 'events' results
        """
        results = {
            "images": [],
            "events": [],
            "unified": [],
        }

        # Search images
        if images is not None:
            results["images"] = self.search.search(query, top_k=10)

        # Search video events
        if events is not None:
            results["events"] = self.search.search_video_events(query, events, top_k=10)

        # Unified ranking
        all_results = []
        for r in results["images"]:
            all_results.append(
                {
                    "type": "image",
                    "score": r["score"],
                    "caption": r["caption"],
                    "path": r.get("path", ""),
                }
            )
        for r in results["events"]:
            all_results.append(
                {
                    "type": "video_event",
                    "score": r["score"],
                    "description": r["description"],
                    "frame": r["frame_index"],
                }
            )

        all_results.sort(key=lambda x: x["score"], reverse=True)
        results["unified"] = all_results[:10]

        return results


def test_image_search():
    """Test function for standalone testing."""
    logger.info("Testing SemanticImageSearch...")

    search = SemanticImageSearch()

    # Add test images
    test_images = [
        ("test1.jpg", "A cute dog playing in the snow"),
        ("test2.jpg", "A beautiful sunset over the ocean"),
        ("test3.jpg", "A busy city street with people"),
    ]

    for path, caption in test_images:
        search.add_image(path, caption)

    # Test search
    results = search.search("a pet dog", top_k=2)
    logger.info(f"Search results for 'a pet dog': {results}")

    # Test events
    test_events = [
        {
            "frame_index": 10,
            "description": "Sudden motion detected",
            "surprise_score": 0.8,
        },
        {
            "frame_index": 25,
            "description": "Normal stable scene",
            "surprise_score": 0.2,
        },
        {
            "frame_index": 50,
            "description": "Significant change in frame",
            "surprise_score": 0.9,
        },
    ]

    event_results = search.search_video_events(
        "something unexpected happened", test_events
    )
    logger.info(f"Event search results: {event_results}")

    return search


if __name__ == "__main__":
    test_image_search()
