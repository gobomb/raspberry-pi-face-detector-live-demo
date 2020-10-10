#!/bin/bash

docker run -d --name=detector --net=host \
	-e NODE_HOST="10.10.67.2" \
	registry.local/face_recognition_detector:sixsq