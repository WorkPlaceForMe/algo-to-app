image="graymatics1/anpr"
container="anpr-server"

docker stop $container
docker rm $container
docker run --gpus all \
    --restart always \
    --name $container \
    --network dev \
    --entrypoint "/bin/bash" \
    -d \
    -w /home/src \
    $image \
    -c "/home/src/server/run-serveronly.sh"
