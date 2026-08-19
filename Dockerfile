# One image, two entrypoints.
#
# Default command: the serving surface, because that is the contract Vertex AI checks when
# it registers a custom container -- an image whose default command does not answer the
# health route cannot become a model version.
#
# The Cloud Run Job overrides the command (see iaac/cloud_run.tf) and runs the Dagster
# inference job instead. Two images would be two dependency sets and two things to keep in
# step for no gain: the scoring code is the same code.

# Pinned by digest, not by tag. `python:3.14-slim` is a moving target: the same git SHA
# would build a different image next month, which quietly breaks the one promise the
# tagging scheme makes -- that `image_tag=<sha>` names a rebuildable artefact. The tag is
# kept alongside the digest so a human can read what this is; the digest is what resolves.
# Refresh with: docker buildx imagetools inspect python:3.14-slim --format '{{.Manifest.Digest}}'
FROM python:3.14-slim@sha256:ce40764625a4ff50df3548277632e7f96c4e77fe75fa848aae9885476e7df5a4

# LightGBM's wheel links against libgomp (OpenMP) and the slim image does not carry it:
# `import lightgbm` dies with "libgomp.so.1: cannot open shared object file". It is the
# only system package this image needs, and it is not optional -- it is the model.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv resolves from the committed uv.lock, so the image's dependency set is the one CI
# tested rather than whatever PyPI serves at build time. Pinned by digest for the same
# reason as the base image, and because `:latest` on the resolver is the one dependency
# that could change *how* the lockfile is interpreted. uv 0.12.5.
COPY --from=ghcr.io/astral-sh/uv@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1 /uv /usr/local/bin/uv

# The interpreter's bundled pip and setuptools are dead weight here: the application runs
# from /app/.venv, which uv builds from the lockfile, and uv needs neither. Left in place
# they were the *only* remaining HIGH findings in an image scan -- pip's vendored msgpack
# and setuptools, in code nothing in this image ever executes. Removing them is cheaper
# than triaging a advisory feed for packages the runtime does not import.
#
# Checked after removal: the serving app imports. The three Dagster code locations have NOT
# yet been loaded from this image -- verify before merging.
RUN python -m pip uninstall --yes pip setuptools \
    && rm -rf /usr/local/lib/python3.14/site-packages/pip \
              /usr/local/lib/python3.14/site-packages/pip-*.dist-info

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    # Without this the resolver's download cache is baked into the layer: tens of megabytes
    # of wheels the runtime never opens, and every one of them is a package an image scan
    # finds and attributes to this image. The venv is the artifact; the cache is scaffolding.
    UV_NO_CACHE=1 \
    # An ephemeral Dagster instance: the job runs once and exits, and a run history that
    # dies with the container is honest about that. Run metadata that matters goes to
    # Vertex Experiments and prediction_logs, both of which outlive the container.
    #
    # Under /tmp because that is the one path Cloud Run guarantees is writable. Dagster
    # refuses to start if DAGSTER_HOME names a directory that does not exist, and Cloud
    # Run mounts /tmp as a fresh tmpfs -- so the directory is created by the job's own
    # command at startup (iaac/cloud_run.tf), not here. This mkdir is for `docker run`.
    DAGSTER_HOME=/tmp/dagster

# The process needs no root privilege: it reads a model from GCS, runs BigQuery queries and
# writes to /tmp. Running as root anyway means a container escape starts from uid 0 for
# nothing gained. Fixed uid so the ownership below is meaningful regardless of the base
# image's /etc/passwd.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

# Owned by appuser because a local `docker run` uses this directory as DAGSTER_HOME. Under
# Cloud Run it is shadowed by a fresh tmpfs and the job's own command recreates it -- as
# appuser, which works because Cloud Run mounts /tmp world-writable.
RUN mkdir -p /tmp/dagster && chown appuser:appuser /tmp/dagster

WORKDIR /app

# Dependencies first, in their own layer: they change on a lockfile edit, the source
# changes on every commit, and the image tag is a git SHA -- so this layer is the
# difference between a 20-second build and a 4-minute one.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src/ ./src/
# Not incidental data files. `config/` carries the admission rules and the gate thresholds,
# `references/` carries the feature contract whose fingerprint the scoring job checks the
# model against, and `schemas/` carries the pinned BigQuery schemas. All three are read by
# relative path from the working directory.
COPY config/ ./config/
COPY references/ ./references/
COPY schemas/ ./schemas/

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}"

# The commit this image was built from, so a container can stamp its own provenance onto
# the models it trains and the rows it scores without carrying a .git directory. Declared
# here, after the expensive layers, because it changes on every commit and anything above
# it would be rebuilt from scratch each time. Defaults to empty: an unstamped local build
# reports `unknown` rather than lying about which commit it is.
# See src/fraud_detection/registry/provenance.py.
ARG GIT_SHA=""
ENV GIT_SHA=${GIT_SHA}

# After the last build step that needs to write to /app. Everything below runs unprivileged.
USER appuser

# Vertex sets AIP_HTTP_PORT; Cloud Run sets PORT. Both default to 8080, and app.py reads
# AIP_HTTP_PORT itself -- this is only the fallback for a bare `docker run`.
EXPOSE 8080

CMD ["sh", "-c", "uvicorn fraud_detection.serving.app:app --host 0.0.0.0 --port ${AIP_HTTP_PORT:-${PORT:-8080}}"]
