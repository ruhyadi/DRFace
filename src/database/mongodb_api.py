"""
MongoDB class for API purposes.
"""

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

from fastapi.encoders import jsonable_encoder
from omegaconf import DictConfig

from src.database.mongodb_base import MongoDBBase
from src.schema.user_schema import UserSchema
from src.schema.face_recognition_schema import FaceEmbeddingSchema
from src.utils.math import find_cosine_distance
from src.utils.exception import APIExceptions
from src.utils.logger import get_logger

exception = APIExceptions()
log = get_logger("mongodb_api")


class MongoDBAPI(MongoDBBase):
    """MondoDB class for API purposes."""

    def __init__(self, cfg: DictConfig) -> None:
        """Constructor.

        Args:
            cfg (DictConfig): MongoDB configuration
        """
        self.cfg = cfg
        super().__init__(
            host=self.cfg.database.mongodb.host,
            port=self.cfg.database.mongodb.port,
            user=self.cfg.database.mongodb.user,
            password=self.cfg.database.mongodb.password,
            db=self.cfg.database.mongodb.db,
        )

    async def get_user(self, username: str) -> UserSchema:
        """Get user information from MongoDB.

        Args:
            username (str): username

        Returns:
            dict: user information

        Raises:
            exception.NotFound: user does not exist
        """
        user = self.find_one("users", {"username": username})
        if user is None:
            raise exception.NotFound(f"User {username} does not exist")
        return UserSchema(**user)

    async def insert_user(self, user: UserSchema) -> UserSchema:
        """Insert user information to MongoDB.

        Args:
            user (UserSchema): user information

        Returns:
            UserSchema: user information

        Raises:
            exception.Conflict: user already exists
        """
        # check if user exists
        if await self.check_user(user.username):
            raise exception.Conflict(f"User {user.username} already exists")

        # insert user
        user = jsonable_encoder(user)
        result = self.insert_one("users", user)
        if result is None:
            raise exception.InternalServerError("Error inserting user")
        log.log(22, f"User {user['username']} {user['_id']} inserted successfully")
        return UserSchema(**user)

    async def check_user(self, username: str) -> bool:
        """Check if user exists in MongoDB.

        Args:
            username (str): username

        Returns:
            bool: True if user exists, False otherwise
        """
        user = self.find_one("users", {"username": username})
        return True if user is not None else False

    async def insert_face(
        self, face_embd: FaceEmbeddingSchema, is_update: bool = False
    ) -> FaceEmbeddingSchema:
        """
        Insert face embedding to MongoDB.

        Args:
            name (str): name
            embedding (np.ndarray): face embedding
            is_update (bool, optional): update flag. Defaults to False.

        Raises:
            exception.Conflict: face embedding already exists
        """
        # check if face embedding exists
        if await self.check_face_database(face_embd.name) and not is_update:
            raise exception.Conflict(
                f"Face embedding for {face_embd.name} already exists"
            )

        face_embd = jsonable_encoder(face_embd)
        result = self.insert_one("faces", face_embd)
        if result is None:
            raise exception.InternalServerError("Error inserting face embedding")
        log.log(22, f"Face embedding for {face_embd['name']} inserted successfully")
        return FaceEmbeddingSchema(**face_embd)

    async def check_face_database(self, name: str) -> bool:
        """
        Check if name exists in face embedding database.

        Args:
            name (str): name

        Returns:
            bool: True if face embedding exists, False otherwise
        """
        name = self.find_one("faces", {"name": name})
        return True if name is not None else False

    async def find_face(
        self,
        face_embd: FaceEmbeddingSchema,
        min_dist_thres: float = 0.5,
        dist_method: str = "cosine",
    ) -> FaceEmbeddingSchema:
        """
        Find name of face embedding in face embedding database with KNN.

        Args:
            face_embd (FaceEmbeddingSchema): face embedding

        Returns:
            FaceEmbeddingSchema: face embedding

        Raises:
            exception.NotFound: face embedding does not exist
        """
        gt_face_embd = await self.load_gt_embeddings()
        if len(gt_face_embd) == 0:
            raise exception.NotFound("Face embedding database is empty")

        # find nearest neighbor
        dist = []
        for gt_embd in gt_face_embd:
            if dist_method == "cosine":
                dist.append(
                    find_cosine_distance(face_embd.embedding, gt_embd["embedding"])
                )
        min_dist = min(dist)
        min_dist_idx = dist.index(min_dist)
        if min_dist > min_dist_thres:
            raise exception.NotFound("Face not matched with any face in database")
        return FaceEmbeddingSchema(**gt_face_embd[min_dist_idx])

    async def load_gt_embeddings(self) -> list:
        """
        Load all face ground truth embeddings from MongoDB.

        Returns:
            list: list of face embeddings
        """
        return self.find_many("faces", {})
