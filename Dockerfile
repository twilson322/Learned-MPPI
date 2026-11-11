# CPU-only environment for the tests and the full pipeline.
#   docker build -t learned-mppi .
#   docker run --rm learned-mppi                                   # unit tests
#   docker run --rm -v $PWD/results:/src/results learned-mppi python3 scripts/run_experiments.py --stage all
FROM python:3.12-slim
WORKDIR /src
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python3", "-m", "pytest", "-q", "tests"]
