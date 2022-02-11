image="graymatics1/face"
container="fr-server"

echo $container started.

docker stop $container
docker rm $container
docker run --gpus all \
    -d \
    --restart always \
    --name $container \
    -v ~/dev/face/src:/home/src \
    --network dev \
    --entrypoint "/bin/bash" \
    -w /home/src/server \
    $image \
    -c "/home/src/server/run.sh"

    #-v /home/ubuntu/dev/helmet/src/server:/home/src/server \
    #-ti \
