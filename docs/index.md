# Welcome to DRFace Framework

A simple framework designed for face recognition purposes. This framework is built on top of FastAPI, MongoDB and OpenVINO Model Server so it's ready to use right away. This page will give you a short quickstart demonstration of the DRFace framework. For Documentation please refer to the Reference.

## Quickstart
We assume that you have already installed [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/).

Build the docker images:
```bash
docker build -t ruhyadi/drface:version .
```

Create a `.env` file:
```bash
cp .env.example .env
```

Run docker container with docker-compose:
```bash
docker-compose up -d
```

## API Documentation
The API documentation is available at [http://localhost:4540](http://localhost:4540).


## Reference
- [FaceNet Pytorch](https://github.com/timesler/facenet-pytorch)
- [deepface](https://github.com/serengil/deepface)
- [insightface](https://github.com/deepinsight/insightface)