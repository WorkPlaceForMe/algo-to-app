image="broadcast:0.0.2"
container="broadcast"
docker stop $container
docker rm $container
docker run --gpus all \
    --restart always \
    -d \
    -v ~/hydrabad:/home/src \
    --name $container \
    --network dev \
    -p 8091:8090 \
    -w /home/src/ \
    --entrypoint "/bin/bash" \
    $image \
    -c "ffserver -f server.conf"

