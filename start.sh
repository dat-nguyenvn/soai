#!/usr/bin/env bash
xhost +local:docker

sudo docker run --gpus all -it --privileged --ipc=host --ulimit memlock=-1 \
 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY\
 -v /tmp/.docker.xauth:/tmp/.docker.xauth\
 -e XAUTHORITY=/tmp/.docker.xauth\
 -v /home/boss/mypc/STOCK:/home/src/stock \
--name STOCK d936dd218d4f
