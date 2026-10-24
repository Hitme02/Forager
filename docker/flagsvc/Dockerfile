# Minimal, auditable image for the deliberately-vulnerable Tier-B research fixture.
# Standard library only -- no third-party packages, no network fetch at build or run time.
FROM python:3.12-slim
WORKDIR /app
COPY app.py /app/app.py
# Drop to a non-root user; the fixture needs nothing privileged.
RUN useradd -m fixture && chown -R fixture /app
USER fixture
EXPOSE 80
CMD ["python", "/app/app.py"]
