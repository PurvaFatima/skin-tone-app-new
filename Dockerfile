FROM python:3.9-slim AS builder 
WORKDIR /app 
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html 
ECHO is on.
FROM python:3.9-slim 
WORKDIR /app 
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages 
COPY --from=builder /usr/local/bin /usr/local/bin 
COPY . . 
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "$PORT"] 
