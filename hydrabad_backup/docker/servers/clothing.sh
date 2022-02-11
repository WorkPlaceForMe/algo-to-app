image="graymatics1/clothing1"
container="test_clothing"

echo $container started.

docker stop $container
docker rm $container
docker run --gpus all \
    --restart always \
    --name $container \
    --network dev \
    --entrypoint "/bin/bash" \
    -w "/home/Quantela/ClothingAttributes" \
    -d \
    $image \
    -c "/home/Quantela/ClothingAttributes/run_gil.sh"

