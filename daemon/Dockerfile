FROM registry.local/face_recognition-base:sixsq
WORKDIR /root
ADD daemon/daemon.py ./daemon/daemon.py
ADD rpc ./rpc
ENTRYPOINT ["python3", "/root/daemon/daemon.py"]