from typing import List, Dict, Any
import os
import pandas as pd
from utils.logging import get_logger

logger = get_logger(__name__)

class ChunkManager:
    """
    A class to handle the chunking of different file types.
    It splits large documents into smaller, semantically meaningful chunks
    to prepare them for vector database indexing.
    """
    def __init__(self):
        logger.info("ChunkManager initialized.")

    def _split_text_by_delimiter(self, text: str, delimiter: str = "\n\n") -> List[str]:
        """
        Splits text by a given delimiter and cleans up the resulting chunks.
        """
        chunks = text.split(delimiter)
        # Clean up chunks by stripping whitespace and removing empty strings
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def create_chunks_from_file(self, file_path: str) -> List[str]:
        """
        Reads a file from the given path and creates chunks from its content.
        
        Args:
            file_path (str): The path to the file to be chunked.
            
        Returns:
            A list of text chunks.
            
        Raises:
            ValueError: If the file type is not supported or the file is empty.
        """
        if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
            logger.error(f"File not found or is empty: {file_path}")
            raise ValueError("Input file is not found or is empty.")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        content = ""

        logger.info(f"Reading content from file: {file_path}")
        try:
            if file_extension in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif file_extension == '.csv':
                # For CSV, we can concatenate the content of all rows/columns,
                # or handle it based on specific needs. Here, we'll
                # just read and convert the whole file to a string.
                df = pd.read_csv(file_path)
                content = df.to_string(index=False)
            else:
                raise ValueError(f"Unsupported file type for chunking: {file_extension}")
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise ValueError(f"Failed to read file for chunking: {e}")

        logger.info(f"Creating chunks from content.")
        
        # A simple, universal approach is to split by paragraphs or double newlines.
        chunks = self._split_text_by_delimiter(content)

        if not chunks:
            logger.warning("No chunks were created. The file might not contain recognizable text.")
            return []

        return chunks