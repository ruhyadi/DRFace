"""Client for FaceNet recognizer."""

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

from src.engine.recognizer.recognizer_base import FaceRecognizerBase
from src.utils.logger import get_logger

log = get_logger("facenet_ovms")


class FaceNetRecognizer(FaceRecognizerBase):
    """FaceNet OVMS client."""

    def __init__(
        self,
        model_name: str = "facenet",
        model_version: str = "1",
        protocol: str = "grpc",
        host: str = "drface-engine-facenet",
        grpc_port: int = 4551,
        http_port: int = 4552,
    ) -> None:
        """
        Initialize FaceNet model.

        Args:
            model_name (str, optional): Model name. Defaults to "facenet".
            model_version (str, optional): Model version. Defaults to "1".
            protocol (str, optional): Protocol. Defaults to "grpc".
            host (str, optional): Host. Defaults to "drface-engine-facenet".
            grpc_port (int, optional): GRPC port. Defaults to 4551.
            http_port (int, optional): HTTP port. Defaults to 4552.

        """
        super().__init__(
            model_name=model_name,
            model_version=model_version,
            protocol=protocol,
            host=host,
            grpc_port=grpc_port,
            http_port=http_port,
        )
        self.input_shape = (160, 160)


if __name__ == "__main__":
    """Debugging."""
    import time
    import cv2

    face = cv2.imread("tmp/sample001_cropped.jpg")
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    net = FaceNetRecognizer(host="localhost")

    start = time.time()
    for _ in range(100):
        response = net.get_embedding(face)
    end = time.time()

    log.info(f"Response time: {(end - start)/100 * 1000:.2f} ms")