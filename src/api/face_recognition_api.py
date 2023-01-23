"""Face recognition API."""

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

import io
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, Depends, FastAPI, File, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from omegaconf import DictConfig
from PIL import Image

from src.api.base_api import BaseAPI
from src.engine.detector.ssd_detector import SSDFaceDetector
from src.engine.recognizer.arcface_recognizer import ArcFaceRecognizer
from src.engine.recognizer.facenet_recognizer import FaceNetRecognizer
from src.schema.auth_schema import CurrentUser
from src.schema.enums_schema import DetectionModel, RecognitionModel
from src.schema.face_detection_schema import FaceDetectionRequest, FaceDetectionResponse
from src.schema.face_recognition_schema import (
    FaceEmbeddingSchema,
    FaceRecognitionRequest,
    FaceRecognitionResponse,
)
from src.schema.log_schema import LogSchema
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
        self.ssd = SSDFaceDetector()
        self.ssd.build_model()

        self.facanet = FaceNetRecognizer(
            model_name=self.cfg.engine.facenet.model_name,
            model_version=self.cfg.engine.facenet.model_version,
            protocol=self.cfg.engine.facenet.protocol,
            host=self.cfg.engine.facenet.host,
            grpc_port=self.cfg.engine.facenet.grpc_port,
            http_port=self.cfg.engine.facenet.http_port,
        )
        self.arcface = ArcFaceRecognizer(
            model_name=self.cfg.engine.arcface.model_name,
            model_version=self.cfg.engine.arcface.model_version,
            protocol=self.cfg.engine.arcface.protocol,
            host=self.cfg.engine.arcface.host,
            grpc_port=self.cfg.engine.arcface.grpc_port,
            http_port=self.cfg.engine.arcface.http_port,
        )

    def setup(self) -> None:
        """Setup API Endpoints."""

        @self.router.post(
            "/api/face/register",
            tags=["face"],
            summary="Register face",
            description="Register face",
            dependencies=[Depends(self.bearer_auth)],
            response_model=FaceDetectionResponse,
            response_model_exclude=["id"],
        )
        async def face_register(
            request: Request,
            request_form: FaceDetectionRequest = Depends(),
            image: UploadFile = File(...),
            current_user: CurrentUser = Depends(self.bearer_auth),
        ):
            """
            Register face.

            Args:
                request_form (FaceDetectionRequest): Name of the person.
                image (UploadFile): Image file.

            Raise:
                exceptions.NotFound: No face detected.
                exceptions.BadRequest: Multiple faces detected.
            """
            start_request = t.now_iso(utc=True)
            request_id = str(uuid.uuid4())
            name = request_form.name.lower().replace(" ", "_")
            detection_model = request_form.detection_model
            recognition_model = request_form.recognition_model

            log.log(
                24, f"Request from {request.client.host} to register face of {name}"
            )

            # get user object
            user = await self.mongodb.get_user(current_user.username)

            # preprocess image
            img = await self.preprocess_raw_img(await image.read())

            # detect face
            if detection_model == DetectionModel.ssd:
                face_dets = self.ssd.detect_face(img)
            if not face_dets:
                raise exceptions.NotFound("No face detected")
            if len(face_dets) > 1:
                raise exceptions.BadRequest("Multiple faces detected")

            # register face
            if recognition_model == RecognitionModel.facenet:
                face_embd = self.facanet.get_embedding(face_dets[0].face)
            elif recognition_model == RecognitionModel.arcface:
                face_embd = self.arcface.get_embedding(face_dets[0].face)
            face_embd = FaceEmbeddingSchema(
                user_id=str(user.id),
                name=name,
                detection_model=detection_model,
                recognition_model=recognition_model,
                embedding=face_embd,
            )
            face_embd = await self.mongodb.insert_face(face_embd, is_update=True)

            end_request = t.now_iso(utc=True)
            request_time = t.diff(start_request, end_request)

            # API response
            response = FaceDetectionResponse(
                request_id=request_id,
                timestamp=start_request,
                status="success",
                detection_model=detection_model,
                recognition_model=recognition_model,
                name=name,
            )

            # logging to database
            logs = LogSchema(
                user_id=str(user.id),
                request_id=request_id,
                timestamp=start_request,
                request_type="face_register",
                request_data=jsonable_encoder(request_form),
                response_data=jsonable_encoder(response),
                response_time=request_time,
            )
            await self.mongodb.insert_log(logs)

            log.log(
                24,
                f"Request to register face of {name} completed in {request_time} ms",
            )

            return response

        @self.router.post(
            "/api/face/recognize",
            tags=["face"],
            summary="Recognize face",
            description="Recognize face",
            dependencies=[Depends(self.bearer_auth)],
            response_model=FaceRecognitionResponse,
            response_model_exclude=["id"],
        )
        async def face_recognize(
            request: Request,
            request_form: FaceRecognitionRequest = Depends(),
            image: UploadFile = File(...),
            current_user: CurrentUser = Depends(self.bearer_auth),
        ) -> FaceRecognitionResponse:
            """
            Recognize face.

            Args:
                image (UploadFile): Image file.

            Return:
                FaceRecognitionResponse: Face recognition response.

            Raise:
                exceptions.NotFound: No face detected.
                exceptions.BadRequest: Multiple faces detected.
            """
            start_request = t.now_iso(utc=True)
            request_id = str(uuid.uuid4())
            detection_model = request_form.detection_model
            recognition_model = request_form.recognition_model
            log.log(24, f"Request from {request.client.host} to recognize face")

            # get user object
            user = await self.mongodb.get_user(current_user.username)

            # preprocess image
            img = await self.preprocess_raw_img(await image.read())

            # detect face
            if detection_model == DetectionModel.ssd:
                face_dets = self.ssd.detect_face(img)
            if not face_dets:
                raise exceptions.NotFound("No face detected")
            if len(face_dets) > 1:
                raise exceptions.BadRequest("Multiple faces detected")

            # recognize face
            ground_truths = await self.mongodb.get_face_gts(
                user_id=str(user.id), recognition_model=recognition_model
            )
            if recognition_model == RecognitionModel.facenet:
                preds = self.facanet.predict(
                    face=face_dets[0].face,
                    ground_truths=ground_truths,
                    dist_method=self.cfg.engine.recognizer.dist_method,
                    dist_threshold=self.cfg.engine.recognizer.min_dist_thres,
                )
            elif recognition_model == RecognitionModel.arcface:
                preds = self.arcface.predict(
                    face=face_dets[0].face,
                    ground_truths=ground_truths,
                    dist_method=self.cfg.engine.recognizer.dist_method,
                    dist_threshold=self.cfg.engine.recognizer.min_dist_thres,
                )

            end_request = t.now_iso(utc=True)
            request_time = t.diff(start_request, end_request)

            # API response
            response = FaceRecognitionResponse(
                request_id=request_id,
                timestamp=start_request,
                status="success",
                detection_model=detection_model,
                recognition_model=recognition_model,
                name=preds.name,
                distance=round(preds.distance, 6),
                dist_method=self.cfg.engine.recognizer.dist_method,
            )

            # logging to database
            logs = LogSchema(
                user_id=str(user.id),
                request_id=request_id,
                timestamp=start_request,
                request_type="face_recognize",
                request_data=jsonable_encoder(request_form),
                response_data=jsonable_encoder(response),
                response_time=request_time,
            )
            await self.mongodb.insert_log(logs)

            log.log(
                24,
                f"Request to recognize face completed in {t.diff(start_request, end_request)}",
            )

            return response

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
