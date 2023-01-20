"""
API exceptions module.
"""

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

from fastapi import HTTPException, status

from src.utils.logger import get_logger

log = get_logger("exceptions")


class APIExceptions:
    """API exceptions class."""

    def __init__(self) -> None:
        """Initialize api exceptions."""
        pass

    def NotFound(self, detail: str = "Not Found") -> HTTPException:
        """
        Handle 404 error (Not Found).

        Args:
            detail (str): Error detail

        Returns:
            HTTPException: HTTPException object

        Examples:
            >>> raise exceptions.NotFound("Not Found")
            2022-01-03 12:00:00,000 [ERROR] 404: Not Found
        """
        log.error(f"404: {detail}")
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)

    def InternalServerError(
        self, detail: str = "Internal Server Error"
    ) -> HTTPException:
        """
        Handle 500 error (Internal Server Error).

        Args:
            detail (str): Error detail

        Returns:
            HTTPException: HTTPException object

        Examples:
            >>> raise exceptions.InternalServerError("Internal Server Error")
            2022-01-03 12:00:00,000 [ERROR] 500: Internal Server Error
        """
        log.error(f"500: {detail}")
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail
        )

    def BadRequest(self, detail: str = "Bad Request") -> HTTPException:
        """
        Handle 400 error (Bad Request).

        Args:
            detail (str): Error detail

        Returns:
            HTTPException: HTTPException object

        Examples:
            >>> raise exceptions.BadRequest("Bad Request")
            2022-01-03 12:00:00,000 [ERROR] 400: Bad Request
        """
        log.error(f"400: {detail}")
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

    def Unauthorized(self, detail: str = "Unauthorized") -> HTTPException:
        """
        Handle 401 error (Unauthorized).

        Args:
            detail (str): Error detail

        Returns:
            HTTPException: HTTPException object

        Examples:
            >>> raise exceptions.Unauthorized("Unauthorized")
            2022-01-03 12:00:00,000 [ERROR] 401: Unauthorized
        """
        log.error(f"401: {detail}")
        return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)

    def Forbidden(self, detail: str = "Forbidden") -> HTTPException:
        """
        Handle 403 error (Forbidden).

        Args:
            detail (str): Error detail

        Returns:
            HTTPException: HTTPException object

        Examples:
            >>> raise exceptions.Forbidden("Forbidden")
            2022-01-03 12:00:00,000 [ERROR] 403: Forbidden
        """
        log.error(f"403: {detail}")
        return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)

    def Conflict(self, detail: str = "Conflict") -> HTTPException:
        """
        Handle 409 error (Conflict).

        Args:
            detail (str): Error detail

        Returns:
            HTTPException: HTTPException object

        Examples:
            >>> raise exceptions.Conflict("Conflict")
            2022-01-03 12:00:00,000 [ERROR] 409: Conflict
        """
        log.error(f"409: {detail}")
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)

    def UnprocessableEntity(
        self, detail: str = "Unprocessable Entity"
    ) -> HTTPException:
        """
        Handle 422 error (Unprocessable Entity).

        Args:
            detail (str): Error detail

        Returns:
            HTTPException: HTTPException object

        Examples:
            >>> raise exceptions.UnprocessableEntity("Unprocessable Entity")
            2022-01-03 12:00:00,000 [ERROR] 422: Unprocessable Entity
        """
        log.error(f"422: {detail}")
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail
        )

    def TooManyRequests(self, detail: str = "Too Many Requests") -> HTTPException:
        """
        Handle 429 error (Too Many Requests).

        Args:
            detail (str): Error detail

        Returns:
            HTTPException: HTTPException object

        Examples:
            >>> raise exceptions.TooManyRequests("Too Many Requests")
            2022-01-03 12:00:00,000 [ERROR] 429: Too Many Requests
        """
        log.error(f"429: {detail}")
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=detail
        )

    def ServiceUnavailable(self, detail: str = "Service Unavailable") -> HTTPException:
        """
        Handle 503 error (Service Unavailable).

        Args:
            detail (str): Error detail

        Returns:
            HTTPException: HTTPException object

        Examples:
            >>> raise exceptions.ServiceUnavailable("Service Unavailable")
            2022-01-03 12:00:00,000 [ERROR] 503: Service Unavailable
        """
        log.error(f"503: {detail}")
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail
        )

    def NotImplemented(self, detail: str = "Not Implemented") -> HTTPException:
        """
        Handle 501 error (Not Implemented).

        Args:
            detail (str): Error detail

        Returns:
            HTTPException: HTTPException object

        Examples:
            >>> raise exceptions.NotImplemented("Not Implemented")
            2022-01-03 12:00:00,000 [ERROR] 501: Not Implemented
        """
        log.error(f"501: {detail}")
        return HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=detail)

    def BadGateway(self, detail: str = "Bad Gateway") -> HTTPException:
        """
        Handle 502 error (Bad Gateway).

        Args:
            detail (str): Error detail

        Returns:
            HTTPException: HTTPException object

        Examples:
            >>> raise exceptions.BadGateway("Bad Gateway")
            2022-01-03 12:00:00,000 [ERROR] 502: Bad Gateway
        """
        log.error(f"502: {detail}")
        return HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail)

    def GatewayTimeout(self, detail: str = "Gateway Timeout") -> HTTPException:
        """
        Handle 504 error (Gateway Timeout).

        Args:
            detail (str): Error detail

        Returns:
            HTTPException: HTTPException object

        Examples:
            >>> raise exceptions.GatewayTimeout("Gateway Timeout")
            2022-01-03 12:00:00,000 [ERROR] 504: Gateway Timeout
        """
        log.error(f"504: {detail}")
        return HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=detail)

    def MethodNotAllowed(self, detail: str = "Method Not Allowed") -> HTTPException:
        """
        Handle 405 error (Method Not Allowed).

        Args:
            detail (str): Error detail

        Returns:
            HTTPException: HTTPException object

        Examples:
            >>> raise exceptions.MethodNotAllowed("Method Not Allowed")
            2022-01-03 12:00:00,000 [ERROR] 405: Method Not Allowed
        """
        log.error(f"405: {detail}")
        return HTTPException(
            status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail=detail
        )


if __name__ == "__main__":
    """Debugging."""

    exceptions = APIExceptions()

    raise exceptions.BadRequest("Bad Request")