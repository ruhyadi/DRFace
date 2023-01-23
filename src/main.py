"""Main class for DRFace."""

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

import hydra
from omegaconf import DictConfig

from src.utils.logger import get_logger

log = get_logger("main")


def main_api(cfg: DictConfig) -> None:
    """Main API module."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    from src.api.base_api import BaseAPI
    from src.api.face_recognition_api import FaceRecognitionAPI
    from src.api.gunicorn_runner import GunicornApp

    # from src.api.gunicorn_runner import GunicornApp

    log.info("Starting DRFace API service...")

    app = FastAPI(
        title="DRFace API",
        description="REST API for DRFace",
        version="0.1.0",
        docs_url="/",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.api.middleware.cors.allow_origins,
        allow_credentials=cfg.api.middleware.cors.allow_credentials,
        allow_methods=cfg.api.middleware.cors.allow_methods,
        allow_headers=cfg.api.middleware.cors.allow_headers,
    )

    # API service
    base_api = BaseAPI(cfg)
    face_recognition_api = FaceRecognitionAPI(cfg)

    # setup router
    app.include_router(base_api.router)
    app.include_router(face_recognition_api.router)

    # ISSUE: https://github.com/SeldonIO/seldon-core/issues/2220
    options = {
        "bind": f"{cfg.api.host}:{cfg.api.port}",
        "workers": cfg.api.workers,
        "worker_class": "uvicorn.workers.UvicornWorker",
        "timeout": cfg.api.timeout,
    }
    GunicornApp(app, options).run()


if __name__ == "__main__":
    @hydra.main(config_path=f"{ROOT}/config", config_name="main", version_base=None)
    def main(cfg: DictConfig) -> None:
        """Main function for the application."""
        if cfg.mode == "api":
            main_api(cfg)

    main()