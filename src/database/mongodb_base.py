"""
MongoDB base/parent class.
"""

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

from typing import Optional

import pymongo
from pymongo.results import InsertOneResult


from src.utils.logger import get_logger

log = get_logger("mongodb_base")


class MongoDBBase:
    """MongoDB base/parent class."""

    def __init__(self, host: str, port: int, user: str, password: str, db: str) -> None:
        """Constructor.

        Args:
            host (str): MongoDB host
            port (str): MongoDB port
            user (str): MongoDB user
            password (str): MongoDB password
            db (str): MongoDB database
        """
        self.host = host
        self.port = int(port)
        self.user = user
        self.password = password
        self.db = db

    def connect(self) -> None:
        """Connec to MongoDB."""
        try:
            self.client = pymongo.MongoClient(
                host=self.host,
                port=self.port,
                username=self.user,
                password=self.password,
                serverSelectionTimeoutMS=2000,
            )
            self.client.server_info()
            self.db = self.client[self.db]
            log.log(22, f"Connected to MongoDB: {self.host}:{self.port}")
        except pymongo.errors.ServerSelectionTimeoutError as e:
            log.error(f"Error connecting to MongoDB: {e}")

    def disconnect(self) -> None:
        """Disconnect from MongoDB."""
        self.client.close()
        log.log(22, f"Disconnected from MongoDB: {self.host}:{self.port}")

    def insert_one(self, database: str, collection: str, data: dict) -> Optional[InsertOneResult]:
        """Insert one document into MongoDB.

        Args:
            database (str): MongoDB database
            collection (str): MongoDB collection
            data (dict): Data to insert

        Returns:
            Optional[InsertOneResult]: InsertOneResult object

        Examples:
            >>> data = {"name": "Didi Ruhyadi", "age": 23}
            >>> insert_one("mydb", "users", data)
            2022-01-03 12:00:00,000 [DEBUG] Inserted document: 5ff1e1c1b3c1b4b9b8b8b8b8
        """
        document = self.db[database][collection].insert_one(data)
        log.debug(f"Inserted document: {document.inserted_id}")
        return document

    def insert_many(self, database: str, collection: str, data: list) -> list[InsertOneResult]:
        """Insert many documents into MongoDB.

        Args:
            database (str): MongoDB database
            collection (str): MongoDB collection
            data (list): Data to insert

        Returns:
            list[InsertOneResult]: List of InsertOneResult objects

        Examples:
            >>> data = [
            ...     {"name": "Didi Ruhyadi", "age": 23},
            ...     {"name": "Sherlin Regiena", "age": 23},
            ... ]
            >>> insert_many("mydb", "users", data)
            2022-01-03 12:00:00,000 [DEBUG] Inserted 2 documents
        """
        documents = self.db[database][collection].insert_many(data)
        log.debug(f"Inserted {len(documents.inserted_ids)} documents")
        return documents

if __name__ == "__main__":

    import hydra

    @hydra.main(config_path=f"{ROOT}/config", config_name="main", version_base=None)
    def main(cfg):
        mongodb = MongoDBBase(
            host=cfg.database.mongodb.host,
            port=cfg.database.mongodb.port,
            user=cfg.database.mongodb.user,
            password=cfg.database.mongodb.password,
            db=cfg.database.mongodb.db,
        )
        mongodb.connect()
        mongodb.disconnect()

    main()