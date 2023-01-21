"""Face recognition API."""

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

import io

import cv2
import numpy as np
from fastapi import APIRouter, Depends, FastAPI, File, Request, UploadFile
from omegaconf import DictConfig
from PIL import Image

from src.api.base_api import BaseAPI
from src.engine.detector.ssd_detector import SSDFaceDetector
from src.engine.recognizer.facenet_recognizer import FaceNetRecognizer
from src.schema.face_recognition_schema import FaceEmbeddingSchema
from src.utils.exception import APIExceptions
from src.utils.logger import get_logger
from src.utils.timer import Timer

app = FastAPI()

exceptions = APIExceptions()
log = get_logger()
t = Timer()


class FaceRecognitionAPI(BaseAPI):
    """Face recognition API module."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize Face recognition API module."""
        self.cfg = cfg
        super().__init__(cfg)
        self.router = APIRouter()

        self.setup_model()
        self.setup()

    def setup_model(self) -> None:
        """Setup face detection and face recogntiion model."""
        self.face_detection = SSDFaceDetector()
        self.face_detection.build_model()

        self.face_recognition = FaceNetRecognizer()
        self.face_recognition.build_model()

    def setup(self) -> None:
        """Setup API Endpoints."""

        @self.router.post(
            "/api/face/register",
            tags=["face"],
            summary="Register face",
            description="Register face",
            dependencies=[Depends(self.bearer_auth)],
        )
        async def face_register(
            request: Request, name: str, image: UploadFile = File(...)
        ):
            """
            Register face.

            Args:
                name (str): Name of the person.
                image (UploadFile): Image file.

            Raise:
                exceptions.NotFound: No face detected.
                exceptions.BadRequest: Multiple faces detected.
            """
            start_request = t.now_iso(utc=True)
            name = name.lower().replace(" ", "_")
            log.log(
                24, f"Request from {request.client.host} to register face of {name}"
            )

            # preprocess image
            img = await self.preprocess_raw_img(await image.read())

            # detect face
            face_dets = self.face_detection.detect_face(img)
            if not face_dets:
                raise exceptions.NotFound("No face detected")
            if len(face_dets) > 1:
                raise exceptions.BadRequest("Multiple faces detected")

            # register face
            face_embd = self.face_recognition.get_embedding(face_dets[0].face)
            face_embd = FaceEmbeddingSchema(
                name=name,
                embedding=face_embd,
            )
            face_embd = await self.mongodb.insert_face(face_embd)

            end_request = t.now_iso(utc=True)
            log.log(
                24,
                f"Request to register face of {name} completed in {t.diff(start_request, end_request)}",
            )

            return {"message": "Face registered"}

        @self.router.post(
            "/api/face/recognize",
            tags=["face"],
            summary="Recognize face",
            description="Recognize face",
            dependencies=[Depends(self.bearer_auth)],
        )
        async def face_recognize(request: Request, image: UploadFile = File(...)):
            """
            Recognize face.

            Args:
                image (UploadFile): Image file.

            Raise:
                exceptions.NotFound: No face detected.
                exceptions.BadRequest: Multiple faces detected.
            """
            start_request = t.now_iso(utc=True)
            log.log(24, f"Request from {request.client.host} to recognize face")

            # preprocess image
            img = await self.preprocess_raw_img(await image.read())

            # detect face
            face_dets = self.face_detection.detect_face(img)
            if not face_dets:
                raise exceptions.NotFound("No face detected")
            if len(face_dets) > 1:
                raise exceptions.BadRequest("Multiple faces detected")

            # recognize face
            face_embd = self.face_recognition.get_embedding(face_dets[0].face)
            face_embd = FaceEmbeddingSchema(embedding=face_embd)
            face_embd = await self.mongodb.find_face(
                face_embd=face_embd,
                min_dist_thres=self.cfg.engine.recognizer.min_dist_thres,
                dist_method=self.cfg.engine.recognizer.dist_method,
            )

            end_request = t.now_iso(utc=True)
            log.log(
                24,
                f"Request to recognize face completed in {t.diff(start_request, end_request)}",
            )

            return {
                "name": face_embd.name,
            }

        app.include_router(self.router)

    async def preprocess_raw_img(self, raw_img: bytes) -> np.ndarray:
        """
        Preprocess raw image. Convert bytes to numpy array.

        Args:
            raw_img (bytes): Raw image.

        Returns:
            np.ndarray: Preprocessed image.
        """
        img = Image.open(io.BytesIO(raw_img))
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
