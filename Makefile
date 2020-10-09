.PHONY: daemon detector

PWD := $(shell pwd)

daemon:
	docker build -f ${PWD}/daemon/Dockerfile  -t registry.local/face_recognition_daemon:0.0.1 ${PWD}

detector:
	docker build -f detetor/Dockerfile  -t registry.local/face_recognition_detetor:0.0.1 ${PWD}