import pandas as pd
import requests
import sqlite3
from io import BytesIO

class DataLoader:
    """Helper class to load data from various sources"""

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file"""
        return pd.read_csv(file_path)

    def load_excel(self, file_path: str) -> pd.DataFrame:
        """Load data from Excel file"""
        return pd.read_excel(file_path)

    def load_file_object(self, file_obj, file_type: str = "csv") -> pd.DataFrame:
        """Load data from a file-like object (e.g., Streamlit uploader)"""
        if file_type == "csv":
            return pd.read_csv(file_obj)
        elif file_type in ["xlsx", "xls"]:
            return pd.read_excel(file_obj)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def load_from_api(self, url: str) -> pd.DataFrame:
        """Load data from API endpoint (assumes JSON/CSV response)"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Try parsing as JSON first
            try:
                data = response.json()
                return pd.DataFrame(data)
            except ValueError:
                # Fallback to CSV
                return pd.read_csv(BytesIO(response.content))
        except Exception as e:
            raise ValueError(f"Failed to load from API: {e}")

    def load_from_database(self, connection_string: str, query: str) -> pd.DataFrame:
        """Load data from database using connection string and query"""
        # Note: This is a simplified implementation. 
        # In production, you'd use SQLAlchemy or specific drivers based on the connection string.
        # For this workshop/demo, we'll assume it might be a sqlite path or generic SQL handling.
        
        try:
            # Basic SQLite support for demo purposes if connection string is a file path
            if 'sqlite' in connection_string or connection_string.endswith('.db'):
                path = connection_string.replace('sqlite:///', '')
                conn = sqlite3.connect(path)
                df = pd.read_sql(query, conn)
                conn.close()
                return df
            else:
                # Placeholder for other DBs
                raise NotImplementedError("Only SQLite supported in this demo loader")
        except Exception as e:
            raise ValueError(f"Database load failed: {e}")
