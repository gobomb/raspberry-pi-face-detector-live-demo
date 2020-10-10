#!/bin/bash

docker run -d --name="video-daemon" --restart=always --net=host --privileged \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v /root/.Xauthority:/root/.Xauthority \
	-v /run/docker.sock:/run/docker.sock \
	-v /var/lib/docker:/var/lib/docker:ro \
	-e DISPLAY=":0.0" -e QT_X11_NO_MITSHM=1 -e QT_GRAPHICSSYSTEM="native" \
	-e CAPTURE_INTERVAL="60" -e FRONTEND_SVC="frontend_svc:8080" \
	--device="/dev/video0:/dev/video0" \
	registry.local/face_recognition_daemon:sixsq
