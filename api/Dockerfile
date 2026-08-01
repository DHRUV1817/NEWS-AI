# Builds the HTTP service for either Hugging Face Spaces or Render — the two
# free tiers the design targets. Neither is chosen yet, so nothing here is
# specific to one: the port is read from the environment, and the image runs as
# an unprivileged user because Spaces requires uid 1000 and Render does not care.

FROM python:3.12-slim AS builder

# uv resolves from the committed lock file, so the image gets the same versions
# the test suite ran against rather than whatever is newest at build time.
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv

# UV_PROJECT_ENVIRONMENT is what redirects `uv sync` to a chosen path. Setting
# VIRTUAL_ENV alone does not: sync would still build the project's own .venv and
# leave the target empty, which fails at run time as "uvicorn: not found".
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

# Two syncs on purpose. The first resolves third-party dependencies from the
# lock alone, so that layer is rebuilt only when pyproject or uv.lock changes
# rather than on every source edit.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# The second installs newsninja itself. Copying the source onto the path is not
# enough: the service reads its version from package metadata rather than a
# literal, so an uninstalled package fails at startup with
# PackageNotFoundError before it can serve a single request.
#
# --no-editable matters. An editable install writes a link back to this build
# directory, which does not exist in the runtime stage, so the copied virtualenv
# would resolve to nothing and fail with ModuleNotFoundError. This puts a real
# copy in site-packages instead.
COPY newsninja ./newsninja
RUN uv sync --frozen --no-dev --no-editable


FROM python:3.12-slim AS runtime

# uid 1000 is what Hugging Face Spaces runs containers as. Matching it here
# means the filesystem permissions are right on both platforms rather than
# only on the one that happens to be lenient.
RUN useradd --create-home --uid 1000 service

# The virtualenv carries the installed package and its dependencies, so no
# source is copied into the runtime stage and nothing depends on the CWD.
COPY --from=builder --chown=service:service /opt/venv /opt/venv

WORKDIR /home/service/app

# The response cache is SQLite on local disk. Both platforms give ephemeral
# storage, so this is a warm cache for the life of one container and nothing
# more — it is not durable state and must not be treated as such.
RUN mkdir -p /home/service/app/.cache && chown service:service /home/service/app/.cache

USER service

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CACHE_PATH=/home/service/app/.cache/newsninja.sqlite

# 7860 is the port Spaces expects. Render injects its own PORT, which wins.
ENV PORT=7860
EXPOSE 7860

# The factory, not a module-level app: building the application at import time
# would construct Settings — and therefore read credentials — during any import,
# including one that never intends to serve.
#
# Wrapped in a shell so ${PORT} expands at start rather than being baked in,
# and `exec` so uvicorn replaces that shell as PID 1. Without the exec the shell
# holds PID 1, SIGTERM on scale-down never reaches uvicorn, and the platform
# kills the container instead of letting it drain.
#
# One worker on purpose: the token limiter's budget window is per process, so a
# second worker would double the effective spend against a single-key quota
# without either worker knowing.
CMD ["sh", "-c", "exec uvicorn newsninja.api:create_app --factory --host 0.0.0.0 --port ${PORT} --workers 1"]
