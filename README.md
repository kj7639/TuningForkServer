# To Dockerize
open bash terminal, cd to this folder
docker build -t tuningforkserver .
docker run -d -p 8000:8000 --name tuningforkserver tuningforkserver
