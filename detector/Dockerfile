FROM registry.local/face_recognition-base:sixsq
WORKDIR /root
ADD detector/detector.py ./detector/detector.py
ADD rpc ./rpc
ENTRYPOINT ["python3", "/root/detector/detector.py"]