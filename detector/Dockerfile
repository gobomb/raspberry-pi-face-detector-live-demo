FROM registry.local/face_recognition-base:sixsq
WORKDIR /root
ADD detector/daemon/daemon.py ./detector/daemon.py
ADD rpc ./rpc
ENTRYPOINT ["python3", "/root/detector.py"]