image="graymatics1/yolov4"
container="yolov4"

echo $container started.

docker stop $container
docker rm $container
docker run --gpus all \
    -v /home/ubuntu/dev/sociald/src/server:/home/algo/src/server \
    -d \
    --name $container \
    --network dev \
    --entrypoint "/bin/bash" \
    -w /home/src/server \
    $image \
    -c "/home/src/server/run.sh"

    #--network dev \
