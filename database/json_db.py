import json
from pathlib import Path
from typing import Dict, Any, List, Optional

class JSONDatabase:
    """
    A simple class to manage a JSON file as a database.
    It handles reading from and writing to the file, and
    ensures the file exists with initial empty data if not.
    """
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self._initialize_db()

    def _initialize_db(self):
        """
        Ensure the database file exists. If not, create it with an empty dictionary.
        """
        if not self.file_path.exists():
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump({}, f)

    def load(self) -> dict:
        """
        Loads all data from the JSON file.
        Returns the data as a dictionary.
        """
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save(self, data: dict):
        """
        Saves the provided data dictionary to the JSON file.
        """
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def create_record(self, key: str, value: Any):
        """
        Adds or updates a record in the database.
        """
        data = self.load()
        data[key] = value
        self.save(data)
        return {key: value}

    def read_record(self, key: str) -> Optional[Any]:
        """
        Reads a single record from the database.
        Returns the value for the given key, or None if not found.
        """
        data = self.load()
        return data.get(key)

    def read_all(self) -> List[Any]:
        """
        Reads all records from the database.
        Returns the data as a list of values.
        """
        data = self.load()
        return list(data.values())

    def delete_record(self, key: str):
        """
        Deletes a record from the database.
        """
        data = self.load()
        if key in data:
            del data[key]
            self.save(data)