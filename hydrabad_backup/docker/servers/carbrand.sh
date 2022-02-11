image="graymatics1/llscarbrands"
container="carbrand_server"

echo $container started.

docker stop $container
docker rm $container
docker run --gpus all \
    --restart always \
    -d \
    -v ~/dev/llsCarBrands/src/:/home/src/ \
    --name $container \
    --network dev \
    -w /home/src/server/ \
    --entrypoint "/bin/bash" \
    $image \
    -c "/home/src/server/run.sh"
