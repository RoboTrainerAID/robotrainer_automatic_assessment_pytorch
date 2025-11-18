#! /bin/bash

DETACH=""
COMMAND="/bin/bash"

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            echo "Running instance segmentation KLEBERAUPE docker container"
            echo "  -h --help       Show help"
	        echo "  -d --detach     Run container detached"
            echo "  --name          give a name for the container"
            echo "  -i --image      Image name to run"
            exit 0
            ;;
        -d|--detach)
            DETACH="-d"
            shift
            ;;
        --name)
            CONTAINER_NAME="$2"
            shift
            shift
            ;;
        -i|--image)
            IMAGE_NAME="$2"
            shift
            shift
            ;;
        *)
            shift
            ;;
    esac
done

if [ -z "$CONTAINER_NAME" ]; then
    CONTAINER_NAME="iras/project-container"
    echo "No container name provided. Using default: $CONTAINER_NAME"
fi

#echo "Using container name: $CONTAINER_NAME and image: $IMAGE_NAME, detach mode: $DETACH, $COMMAND"

docker run -it --rm --privileged $DETACH \
   --name "$CONTAINER_NAME" \
   --ipc=host \
   --net=host \
   --gpus=all \
   -e DISPLAY=$DISPLAY \
   --env-file project.env \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   -v ~/.Xauthority:/root/.Xauthority \
   -v /etc/timezone:/etc/timezone:ro \
   -v /run/dbus/system_bus_socket:/run/dbus/system_bus_socket:ro \
   -v $PWD/project:/workspace/$CONTAINER_NAME:rw \
   -v $PWD/data/:/data/:ro \
   $IMAGE_NAME \
   $COMMAND
