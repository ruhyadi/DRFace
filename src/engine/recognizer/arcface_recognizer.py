"""Insight face recognizer model."""

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

from src.engine.recognizer.recognizer_base import FaceRecognizerBase
from src.utils.logger import get_logger

log = get_logger("arcface_recognize")

class ArcFaceRecognizer(FaceRecognizerBase):
    """InsightFace OVMS client."""

    def __init__(
        self,
        model_name: str = "arcface",
        model_version: str = "1",
        protocol: str = "grpc",
        host: str = "drface-engine-arcface",
        grpc_port: int = 4553,
        http_port: int = 4554,
    ) -> None:
        """
        Initialize ArcFace model.

        Args:
            model_name (str, optional): Model name. Defaults to "arcface".
            model_version (str, optional): Model version. Defaults to "1".
            protocol (str, optional): Protocol. Defaults to "grpc".
            host (str, optional): Host. Defaults to "drface-engine-insightface".
            grpc_port (int, optional): GRPC port. Defaults to 4553.
            http_port (int, optional): HTTP port. Defaults to 4554.
        """
        super().__init__(
            model_name=model_name,
            model_version=model_version,
            protocol=protocol,
            host=host,
            grpc_port=grpc_port,
            http_port=http_port,
        )
        self.input_shape = (112, 112)
    
if __name__ == "__main__":
    """Debugging."""
    import time
    import cv2

    face = cv2.imread("tmp/sample001_cropped.jpg")
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    net = ArcFaceRecognizer(host="localhost")

    start = time.time()
    response = net.get_embedding(face)
    end = time.time()

    log.info(f"Response: {response}")
    log.info(f"Response time: {(end - start) * 1000:.2f} ms")
    log.info(f"Response shape: {len(response)}")