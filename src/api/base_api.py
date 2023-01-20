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

from src.database.mongodb_api import MongoDBAPI
from src.schema.user_schema import UserSchema, UserRegisterResponse
from src.utils.authentication import Authentication
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

        # authentication
        self.auth = Authentication(
            secret=self.cfg.api.auth.secret_key,
            algorithm=self.cfg.api.auth.algorithm,
            expiration=self.cfg.api.auth.token_expiration,
            encrypt_scheme=self.cfg.api.auth.encrypt_scheme,
        )

        # mongodb database
        self.mongodb = MongoDBAPI(self.cfg)

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
            user = await self.authenticate_user(request.username, request.password)
            access_token = await self.auth.create_access_token(
                data={"sub": user.username}
            )
            return Token(access_token=access_token, token_type="bearer")

        @self.router.post(
            "/api/user/register",
            description="User registration",
            tags=["user"],
            response_model=UserRegisterResponse,
        )
        async def register(user: UserSchema) -> UserRegisterResponse:
            """User registration

            Args:
                user (UserSchema): User object

            Returns:
                UserSchema: User object
            """
            # hash password
            user.password = await self.auth.get_password_hash(user.password)

            # insert user
            user = await self.mongodb.insert_user(user)

            return user

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

    async def authenticate_user(self, username: str, password: str) -> UserSchema:
        """Authenticate user

        Args:
            username (str): Username
            password (str): Password

        Returns:
            UserSchema: User object

        Raises:
            exception.Unauthorized: 401
        """
        user = await self.mongodb.get_user(username=username)
        if not await self.auth.verify_password(password, user.password):
            exception.Unauthorized(detail="Incorrect username or password")

        return user
