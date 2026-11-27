FROM python:3.11-slim

WORKDIR /code

COPY ./backend/requirements.txt /code/requirements.txt

# Install torch CPU-only first (the default PyPI torch is the CUDA variant ~5 GB
# which times out on HF free-tier. CPU-only wheel is ~250 MB and is all we need.)
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (without --upgrade so CPU-only torch is preserved)
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy the rest of the files including backend, models, etc.
COPY . .

# Expose the HF spaces required port
EXPOSE 7860

# Run the FastAPI app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
