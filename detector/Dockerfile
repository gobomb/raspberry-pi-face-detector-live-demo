FROM registry.local/face_recognition-base:sixsq
RUN pip3 install grpcio -i https://pypi.doubanio.com/simple
RUN pip3 install protobuf -i https://pypi.doubanio.com/simple
WORKDIR /root
ADD detector/detector.py ./detector/detector.py
ADD rpc ./rpc
ENTRYPOINT ["python3", "/root/detector/detector.py"]