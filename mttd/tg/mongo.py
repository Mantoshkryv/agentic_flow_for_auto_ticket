# ta/services/mongo.py
import os
import pandas as pd
from pymongo import MongoClient, errors

class MongoService:
    def __init__(self):
        """Initialize MongoDB connection"""
        uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        db_name = os.getenv("MONGO_DB", "mttddb")
        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            self.db = self.client[db_name]
            # Test connection
            self.client.server_info()
        except errors.ServerSelectionTimeoutError as e:
            raise ConnectionError(f"Cannot connect to MongoDB: {e}")

    def fetch_kpi(self):
        """Fetch KPI data"""
        docs = list(self.db["kpi_data"].find())
        return pd.DataFrame(docs) if docs else pd.DataFrame()

    def fetch_sessions(self):
        """Fetch session data"""
        docs = list(self.db["sessions"].find())
        return pd.DataFrame(docs) if docs else pd.DataFrame()

    def fetch_advancetags(self):
        """Fetch AdvanceTags data"""
        docs = list(self.db["advancetags"].find())
        return pd.DataFrame(docs) if docs else pd.DataFrame()

    def insert_tickets(self, tickets: list):
        """Safely insert tickets"""
        if tickets:
            try:
                self.db["tickets"].insert_many(tickets)
            except Exception as e:
                raise RuntimeError(f"Failed to insert tickets: {e}")
