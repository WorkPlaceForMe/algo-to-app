image="graymatics1/violence"
container="violence-server"

echo $container started.

docker stop $container
docker rm $container
docker run --gpus all \
    --restart always \
    -d \
    -v /home/ubuntu/dev/violence/src:/home/src \
    --name $container \
    --network dev \
    --entrypoint "/bin/bash" \
    -w /home/src/server \
    $image \
    -c "/home/src/server/run.sh"
