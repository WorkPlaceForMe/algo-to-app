image="graymatics1/deep_yolo"
container="deep_yolo"
docker stop $container
docker rm $container
docker run --gpus all \
    --restart always \
    -v ~/dev/deep_yolo/src:/home/src \
    -d \
    --name $container \
    --network dev \
    -w /home/src/ \
    --entrypoint "/bin/bash" \
    $image \
    -c "bash /home/src/run.sh"

