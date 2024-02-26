docker build -t debug -f Dockerfile .
docker run --gpus all -v $PWD:/output --rm debug