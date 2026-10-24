# Agent-harness image for the Tier-B live range. Runs ON the isolated (internal) network and
# drives the fixture services by DNS name. Kept minimal: only the dependencies the live flag
# run needs (no torch/numba/mpi4py). Build context is the repo root so it can copy `ffs/`.
FROM python:3.12-slim
WORKDIR /work
COPY docker/harness/requirements-harness.txt /work/requirements-harness.txt
RUN pip install --no-cache-dir -r requirements-harness.txt
COPY ffs /work/ffs
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "ffs.realism.run_tierb", "--live"]
