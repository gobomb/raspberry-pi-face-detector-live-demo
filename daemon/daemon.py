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
       Global = rpc_helper.getVal()
        Global.is_called = True
        print("\rIncomming connection which ID is: {0}".format(request.ID))
        try:
            while not Global.is_exit:
                # Read a single frame from frame list
                frame_process = Global.frame

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
            Global.is_called = False

        finally:
            Global.is_called = False
            print("\rIncomming connection closed which ID: {0}".format(request.ID))

    def DisplayLocations(self, request_iterator, context):
        Global = rpc_helper.getVal()
        Global.is_called = True

        try:
            for message in request_iterator:
                # get face_locations, face_names
                global face_names, face_locations
                face_names = message.Face_names
                face_locations = []
                for i in range(0, len(message.Face_locations)):
                    face_locations.append(tuple(message.Face_locations[i].Loc))

                Global.face_names = face_names
                Global.face_locations = face_locations

            return face_pb2.LocationResponse(
                Status=face_pb2.STATUS_OK,
            )
        except grpc.RpcError as rpc_error_call:
            details = rpc_error_call.details()
            print("err='RPCError DisplayLocations'")
            print("errMore=\"" + details + "\"")
        except Exception as ex:
            traceback.print_exc()
            Global.is_called = False
        finally:
            Global.is_called = False


class helper():
    def setVal(self, Global, ):
        helper.Global = Global

    def getVal(self):
        return helper.Global


rpc_helper = helper()


def serve(Global):
    print("start serving rpc")
    rpc_helper.setVal( Global, )
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    face_pb2_grpc.add_FaceServiceServicer_to_server(FaceService(), grpcServer)

    grpcServer.add_insecure_port("{0}:{1}".format(_HOST, _PORT))
    grpcServer.start()
    print("waiting for incomming connection at {0}:{1}".format(_HOST, _PORT))
    # grpcServer.wait_for_termination()
    while True:
        if Global.is_exit:
            break


def frame_helper(worker_id, read_frame_list, write_frame_list, Global, worker_num):
    print("frame_helper {0}/{1} start".format(worker_id, worker_num))

    while not Global.is_exit:
        # Wait to read
        while Global.read_num != worker_id or Global.read_num != prev_id(Global.buff_num,
                                                                         worker_num):

            # If the user has requested to end the app, then stop waiting for webcam frames
            if Global.is_exit:
                break

            time.sleep(0.01)

        display(worker_id, read_frame_list, write_frame_list, Global, worker_num)


def setFrame(Global,frame):
    Global.frame = frame


def displayResult(Global,frame_process):
    face_names = Global.face_names
    face_locations = Global.face_locations

    # Display the results

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw a box around the face
        cv2.rectangle(frame_process, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame_process, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame_process, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

def capture1(Global):
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)

    # Initialize some variables
    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Rotate 90 degrees
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if process_this_frame:
            setFrame(Global,frame)
            displayResult(Global)

        process_this_frame = not process_this_frame

        cv2.namedWindow("Video", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit any key to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Release handle to the webcam
            video_capture.release()
            cv2.destroyAllWindows()
            os._exit(0)


def run():
    # Global variables
    Global = Manager().Namespace()
    Global.buff_num = 1
    Global.read_num = 1
    Global.write_num = 1
    Global.is_exit = False
    Global.is_called = False
    Global.face_names = []
    Global.face_locations = []
    read_frame_list = Manager().dict()
    write_frame_list = Manager().dict()

    Process(target=capture1, args=(Global,)).start()
    Process(target=serve, args=(  Global,).start())

if __name__ == '__main__':
    # threading.Thread(target=scanDetector).start()
    # capture()
    run()
