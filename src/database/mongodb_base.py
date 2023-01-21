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
from pymongo.results import InsertOneResult, InsertManyResult, UpdateResult


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

        self.connect()

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

    def insert_one(self, collection: str, data: dict) -> Optional[InsertOneResult]:
        """Insert one document into MongoDB.

        Args:
            collection (str): MongoDB collection
            data (dict): Data to insert

        Returns:
            Optional[InsertOneResult]: InsertOneResult object

        Examples:
            >>> data = {"name": "Didi Ruhyadi", "age": 23}
            >>> insert_one("users", data)
            2022-01-03 12:00:00,000 [DEBUG] Inserted document: 5ff1e1c1b3c1b4b9b8b8b8b8
        """
        document = self.db[collection].insert_one(data)
        log.debug(f"Inserted document: {document.inserted_id}")
        return document

    def insert_many(self, collection: str, data: list) -> Optional[InsertManyResult]:
        """Insert many documents into MongoDB.

        Args:
            collection (str): MongoDB collection
            data (list): Data to insert

        Returns:
            InsertManyResult: InsertManyResult objects

        Examples:
            >>> data = [
            ...     {"name": "Didi Ruhyadi", "age": 23},
            ...     {"name": "Sherlin Regiena", "age": 23},
            ... ]
            >>> insert_many("users", data)
            2022-01-03 12:00:00,000 [DEBUG] Inserted 2 documents
        """
        documents = self.db[collection].insert_many(data)
        log.debug(f"Inserted {len(documents.inserted_ids)} documents")
        return documents

    def find_one(self, collection: str, query: dict) -> dict:
        """Find one document from MongoDB.

        Args:
            collection (str): MongoDB collection
            query (dict): Query to find

        Returns:
            dict: Document

        Examples:
            >>> query = {"name": "Didi Ruhyadi"}
            >>> find_one("users", query)
            2022-01-03 12:00:00,000 [DEBUG] Found 1 document
            {'_id': ObjectId('5ff1e1c1b3c1b4b9b8b8b8b8'), 'name': 'Didi Ruhyadi', 'age': 23}
        """
        document = self.db[collection].find_one(query)
        log.debug(f"Found document with id: {document['_id']}") if document else None
        return document

    def find_many(self, collection: str, query: dict) -> list:
        """Find many documents from MongoDB.

        Args:
            collection (str): MongoDB
            query (dict): Query to find

        Returns:
            list: Documents

        Examples:
            >>> query = {"name": "Didi Ruhyadi"}
            >>> find_many("users", query)
            2022-01-03 12:00:00,000 [DEBUG] Found 1 document
            [{'_id': ObjectId('5ff1e1c1b3c1b4b9b8b8b8b8'), 'name': 'Didi Ruhyadi', 'age': 23}]
        """
        documents = list(self.db[collection].find(query))
        log.debug(f"Found {len(documents)} documents") if documents else None
        return documents

    def update_one(self, collection: str, query: dict, data: dict) -> None:
        """Update one document from MongoDB.

        Args:
            collection (str): MongoDB
            query (dict): Query to find
            data (dict): Data to update

        Examples:
            >>> query = {"name": "Didi Ruhyadi"}
            >>> data = {"age": 24}
            >>> update_one("users", query, data)
            2022-01-03 12:00:00,000 [DEBUG] Updated 1 document
        """
        document = self.db[collection].update_one(query, {"$set": data})
        log.debug(f"Updated {collection}/{query} document")

    def update_many(self, collection: str, query: dict, data: dict) -> None:
        """Update many documents from MongoDB.

        Args:
            collection (str): MongoDB
            query (dict): Query to find
            data (dict): Data to update

        Examples:
            >>> query = {"name": "Didi Ruhyadi"}
            >>> data = {"age": 24}
            >>> update_many("users", query, data)
            2022-01-03 12:00:00,000 [DEBUG] Updated 3 documents
        """
        documents = self.db[collection].update_many(query, {"$set": data})
        log.debug(
            f"Updated {collection}/{query} -> {documents.modified_count} documents"
        )

    def delete_one(self, collection: str, query: dict) -> None:
        """Delete one document from MongoDB.

        Args:
            collection (str): MongoDB
            query (dict): Query to find

        Examples:
            >>> query = {"name": "Didi Ruhyadi"}
            >>> delete_one("users", query)
            2022-01-03 12:00:00,000 [DEBUG] Deleted 1 document
        """
        document = self.db[collection].delete_one(query)
        log.debug(f"Deleted {collection}/{query} -> {document.deleted_count} document")

    def delete_many(self, collection: str, query: dict) -> None:
        """Delete many documents from MongoDB.

        Args:
            collection (str): MongoDB
            query (dict): Query to find

        Examples:
            >>> query = {"name": "Didi Ruhyadi"}
            >>> delete_many("users", query)
            2022-01-03 12:00:00,000 [DEBUG] Deleted 3 documents
        """
        documents = self.db[collection].delete_many(query)
        log.debug(
            f"Deleted {collection}/{query} -> {documents.deleted_count} documents"
        )


if __name__ == "__main__":
    """Debugging."""

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

        # # Insert one document
        # document = mongodb.insert_one(
        #     collection="users",
        #     data={"name": "Didi Ruhyadi", "age": 23},
        # )
        # log.info(f"Inserted document: {document.inserted_id}")

        # # Insert many documents
        # documents = mongodb.insert_many(
        #     collection="users",
        #     data=[
        #         {"name": "Didi Ruhyadi", "age": 23},
        #         {"name": "Sherlin Regiena", "age": 23},
        #     ],
        # )
        # log.info(f"Inserted documents: {documents.inserted_ids}")

        # # Find one document
        # document = mongodb.find_one(collection="users", query={"name": "Didi Ruhyadi"})
        # log.info(f"Found document: {document}")

        # # Find many documents
        # documents = mongodb.find_many(collection="users", query={"name": "Didi Ruhyadi"})
        # log.info(f"Found documents: {documents}")

        # # Update one document
        # mongodb.update_one(
        #     collection="users",
        #     query={"name": "Didi Ruhyadi"},
        #     data={"age": 24},
        # )

        # # Update many documents
        # mongodb.update_many(
        #     collection="users",
        #     query={"name": "Didi Ruhyadi"},
        #     data={"age": 24},
        # )

        # Delete one document
        # mongodb.delete_one(collection="users", query={"name": "Didi Ruhyadi"})

    main()
