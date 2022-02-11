image="graymatics1/hydrabad4"
container="hydrabad"
src="/home/ubuntu/hydrabad/src"
vid="/home/ubuntu/Multi_Tenant_App/server/resources"
resources="/home/ubuntu/Multi_Tenant_App/server/resources"
face="/home/ubuntu/hydrabad/face_data"
mysqldb="multi_tenant"
mysqlip="172.18.0.1"
#mysqlip="127.0.0.1"
serverip="40.84.143.162"

docker stop $container
docker rm $container
docker run --gpus all\
    --restart always \
    -d \
    -v $src:/home/src/ \
    -v $vid:/home/videos/ \
    -v $resources:/home/resources/ \
    -v $face:/home/face_data/ \
    -e MYSQL_DB=$mysqldb \
    -e MYSQL_IP=$mysqlip \
    -e SERVER_IP=$serverip \
    --name $container \
    -w /home/src/ \
    --network dev \
    --entrypoint "/bin/bash" \
    $image \
    -c "bash run.sh"
