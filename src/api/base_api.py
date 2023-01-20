"""Base API module."""

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

import secrets

from fastapi import APIRouter, Depends, FastAPI
from fastapi.security import HTTPBasic, HTTPBasicCredentials, OAuth2PasswordRequestForm
from omegaconf import DictConfig

from src.schema.user_schema import UserSchema
from src.schema.auth_schema import CurrentUser, Token
from src.utils.exception import APIExceptions
from src.utils.logger import get_logger
from src.utils.timer import Timer

# api app
app = FastAPI()
basic_auth = HTTPBasic()

exception = APIExceptions()
log = get_logger()
t = Timer()


class BaseAPI:
    """Base API module."""

    def __init__(self, cfg: DictConfig) -> None:
        """Constructor

        Args:
            cfg (DictConfig): Configuration object
        """
        self.cfg = cfg
        self.router = APIRouter()

    def setup(self) -> None:
        """Setup API."""
        
        @self.router.post(
            "/api/user/login",
            description="User login",
            tags=["user"],
            response_model=Token,
        )
        async def login(
            request: OAuth2PasswordRequestForm = Depends(),
        ) -> Token:
            """User login

            Args:
                request (OAuth2PasswordRequestForm): Login request object

            Returns:
                Token: Token object

            Raises:
                HTTPException: 401
            """
            pass

    def basic_auth(self, credentials: HTTPBasicCredentials = Depends(basic_auth)):
        """Basic authentication

        Args:
            credentials (HTTPBasicCredentials, optional): Credentials. Defaults to Depends(basic_auth).

        Raises:
            HTTPException: 401
        """
        correct_username = secrets.compare_digest(
            credentials.username, self.cfg.api.auth.basic.username
        )
        correct_password = secrets.compare_digest(
            credentials.password, self.cfg.api.auth.basic.password
        )
        if not (correct_username and correct_password):
            exception.Unauthorized(detail="Incorrect username or password")