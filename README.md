# To Dockerize
open bash terminal, cd to this folder
docker build -t tuningforkserver .
docker run -d -p 8000:8000 --name tuningforkserver tuningforkserver

don't run with -d if you want to be able to see the terminal output