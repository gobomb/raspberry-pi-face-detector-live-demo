.PHONY: daemon detector

PWD := $(shell pwd)

daemon:
	docker build -f ${PWD}/daemon/Dockerfile  -t registry.local/face_recognition_daemon:0.0.1 ${PWD}

detector:
	docker build -f ${PWD}/detector/Dockerfile  -t registry.local/face_recognition_detector:0.0.1 ${PWD}