FROM python:3.12-slim
WORKDIR /app
COPY intraday_scalper/requirements.txt /app/intraday_scalper/requirements.txt
RUN pip install --no-cache-dir -r /app/intraday_scalper/requirements.txt
COPY . /app
ENV PYTHONPATH=/app
ENV TERMINAL_HOST=0.0.0.0
EXPOSE 8080
CMD ["bash", "start.sh"]
