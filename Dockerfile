FROM python:3.12.1

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

ENV PYTHONPATH="/"

COPY . /code

EXPOSE 8000

ENTRYPOINT ["python", "deploy.py"]

# CMD ["fastapi", "run", "deploy.py", "--host", "0.0.0.0", "--port", "8000"]