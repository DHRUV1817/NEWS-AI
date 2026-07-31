"""Application assembly.

``create_app`` exists because middleware is configured from settings at
construction time. Adding CORS at module scope would read the environment once,
at import, and no later configuration could change it — which makes the
allowlist both untestable and unchangeable after the first import.

There is deliberately no module-level ``app`` instance. Constructing one at
import time would call ``get_settings()`` on any import of anything under
``newsninja.api`` — including a plain test collection pass — and that reads
credentials from the environment or ``.env``. On a machine with no configured
key that turns "import this package" into a crash, and on a machine that does
have one it means merely importing the package pulls a real key into memory.
The server is started as a factory instead:

    uvicorn newsninja.api:create_app --factory
"""

from importlib.metadata import version

from fastapi import FastAPI

from newsninja.api.routes import router
from newsninja.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the service. Pass ``settings`` to override the environment."""
    resolved = settings if settings is not None else get_settings()

    application = FastAPI(
        title="NewsNinja",
        description="Source-grounded news briefings with structured extraction.",
        version=version("newsninja"),
    )
    application.state.settings = resolved
    application.include_router(router)
    return application
