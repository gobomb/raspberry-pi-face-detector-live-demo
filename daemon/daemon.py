import cv2
import numpy as np
from io import BytesIO
import traceback

import grpc
import sys
import os
from concurrent import futures
from multiprocessing import Process, Manager, cpu_count, set_start_method
import time
import numpy
import threading
import requests

import platform

sys.path.append("rpc")
import face_pb2
import face_pb2_grpc

_HOST = '0.0.0.0'
_PORT = '9900'
_RPC_WORKER = 2

gframe = []
face_names = []
face_locations = []
exit = False


def encode_frame(pb_frame):
    # numpy to bytes
    nda_bytes = BytesIO()
    np.save(nda_bytes, pb_frame, allow_pickle=False)
    return nda_bytes


def decode_frame(nda_bytes):
    # bytes to numpy
    nda_bytes = BytesIO(nda_proto.ndarray)
    return np.load(nda_bytes, allow_pickle=False)


class FaceService(face_pb2_grpc.FaceServiceServicer):
    def GetFrameStream(self, request, context):
        print("\rIncomming connection which ID is: {0}".format(request.ID))
        try:
            while True:
                frame_process = gframe

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_frame = frame_process[:, :, ::-1]

                byteFrame = encode_frame(rgb_frame).getvalue()

                yield face_pb2.FrameStream(
                    ID=request.ID,
                    Rgb_small_frame=byteFrame,
                    Status=face_pb2.STATUS_OK,
                )

        except Exception as ex:
            traceback.print_exc()

        finally:
            print("\rIncomming connection closed which ID: {0}".format(request.ID))

    def DisplayLocations(self, request_iterator, context):
        try:
            for message in request_iterator:
                # get face_locations, face_names
                global face_names, face_locations
                l_face_names = message.Face_names
                l_face_locations = []
                for i in range(0, len(message.Face_locations)):
                    face_locations.append(tuple(message.Face_locations[i].Loc))

                face_names = l_face_names
                face_locations = l_face_locations

            return face_pb2.LocationResponse(
                Status=face_pb2.STATUS_OK,
            )
        except grpc.RpcError as rpc_error_call:
            details = rpc_error_call.details()
            print("err='RPCError DisplayLocations'")
            print("errMore=\"" + details + "\"")
        except Exception as ex:
            traceback.print_exc()



def serve():
    print("start serving rpc")
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    face_pb2_grpc.add_FaceServiceServicer_to_server(FaceService(), grpcServer)

    grpcServer.add_insecure_port("{0}:{1}".format(_HOST, _PORT))
    grpcServer.start()
    print("waiting for incomming connection at {0}:{1}".format(_HOST, _PORT))
    # grpcServer.wait_for_termination()
    while True:
        if exit:
            break

def displayResult(frame_process):
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw a box around the face
        cv2.rectangle(frame_process, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame_process, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame_process, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

def capture1():
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)

    # Initialize some variables
    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        if process_this_frame:
            global gframe
            gframe = frame
            displayResult(frame)

        process_this_frame = not process_this_frame

        # Display the resulting image
        frame = cv2.flip(frame, 0)
        cv2.imshow('Video', frame)

        # Hit any key to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Release handle to the webcam
            video_capture.release()
            cv2.destroyAllWindows()
            os._exit(0)


def run():
    Process(target=capture1).start()
    serve()

if __name__ == '__main__':
    run()
