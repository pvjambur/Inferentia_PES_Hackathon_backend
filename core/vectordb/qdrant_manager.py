from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid
from typing import List, Dict, Optional, Any
import logging
from config import settings

logger = logging.getLogger(__name__)

class QdrantManager:
    def __init__(self):
        self.client = None
        self.encoder = None
        
    async def initialize(self):
        """Initialize Qdrant client and sentence transformer"""
        try:
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None
            )
            
            # Initialize sentence transformer for embeddings
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Qdrant client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise
            
    async def create_collection(self, collection_name: str, vector_size: int = 384) -> bool:
        """Create a new collection"""
        try:
            collections = self.client.get_collections().collections
            existing_names = [col.name for col in collections]
            
            if collection_name not in existing_names:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {collection_name}")
                return True
            else:
                logger.info(f"Collection {collection_name} already exists")
                return True
                
        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {e}")
            return False
            
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            return False
            
    async def add_chunks(self, collection_name: str, chunks: List[Dict]) -> bool:
        """Add text chunks to collection"""
        try:
            # Ensure collection exists
            await self.create_collection(collection_name)
            
            points = []
            for chunk in chunks:
                # Generate embedding for chunk text
                vector = self.encoder.encode(chunk['text']).tolist()
                
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        'text': chunk['text'],
                        'metadata': chunk.get('metadata', {}),
                        'source': chunk.get('source', ''),
                        'chunk_index': chunk.get('chunk_index', 0)
                    }
                )
                points.append(point)
                
            # Batch upsert points
            self.client.upsert(collection_name, points)
            logger.info(f"Added {len(chunks)} chunks to {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding chunks to {collection_name}: {e}")
            return False
            
    async def search_similar(self, collection_name: str, query: str, limit: int = 5) -> List[Dict]:
        """Search for similar chunks"""
        try:
            # Generate query embedding
            query_vector = self.encoder.encode(query).tolist()
            
            # Search
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True
            )
            
            results = []
            for point in search_result:
                results.append({
                    'id': str(point.id),
                    'text': point.payload.get('text', ''),
                    'metadata': point.payload.get('metadata', {}),
                    'source': point.payload.get('source', ''),
                    'score': float(point.score)
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Error searching in {collection_name}: {e}")
            return []
            
    async def get_collection_info(self, collection_name: str) -> Optional[Dict]:
        """Get collection information"""
        try:
            info = self.client.get_collection(collection_name)
            return {
                'name': collection_name,
                'points_count': info.points_count,
                'vectors_count': info.vectors_count,
                'config': {
                    'distance': info.config.params.vectors.distance.value,
                    'size': info.config.params.vectors.size
                }
            }
        except Exception as e:
            logger.error(f"Error getting collection info for {collection_name}: {e}")
            return None
            
    async def list_collections(self) -> List[str]:
        """List all collections"""
        try:
            collections = self.client.get_collections().collections
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
            
    async def close(self):
        """Close the client connection"""
        if self.client:
            # Qdrant client doesn't need explicit closing
            logger.info("Qdrant client connection closed")