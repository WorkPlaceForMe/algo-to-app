image="graymatics1/weapon2"
container="weapon2-server"

echo $container started.

docker stop $container
docker rm $container
docker run --gpus all \
    --restart always \
    -d \
    -v /home/ubuntu/dev/weapon/src:/home/src \
    --name $container \
    --network dev \
    --entrypoint "/bin/bash" \
    -w /home/src/server \
    $image \
    -c "/home/src/server/run.sh"
