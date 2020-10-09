import http.server as server
from http.server import HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path, PurePosixPath
import face_recognition
import numpy as np
from io import BytesIO
import traceback
import grpc
import os
from multiprocessing import Process, Manager, cpu_count, set_start_method
import sys

sys.path.append("rpc")
import face_pb2
import face_pb2_grpc

# Initialize some variables
_HOST = '192.168.1.109'
_PORT = '9900'
ideal_distance = 0.45


def encode_frame(pb_frame):
    # numpy to bytes
    nda_bytes = BytesIO()
    np.save(nda_bytes, pb_frame, allow_pickle=False)
    return nda_bytes


def decode_frame(ndarray):
    # bytes to numpy
    nda_bytes = BytesIO(ndarray)
    return np.load(nda_bytes, allow_pickle=False)


def find_face(rgb_small_frame):
    # Find all the faces and face encodings in the current frame of video
    face_locations = []
    face_encodings = []
    face_names = []
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if np.mean(face_distances) <= ideal_distance:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
        else:
            name = "Unknown"
        print("find_face {0}/{1}".format(name, face_distances))

        face_names.append(name)
    return face_locations, face_names


def send_message(stub, worker_id):
    for response in stub.GetFrameStream(face_pb2.FrameRequest(ID=str(worker_id))):
        if response.Error:
            print("error when call rpc GetFrameStream")
            print(response.Error)
            return

        face_locations, face_names = find_face(decode_frame(response.Rgb_small_frame))
        locations = []
        for i in range(0, len(face_locations)):
            lc = []
            for j in range(0, len(face_locations[i])):
                lc.append(face_locations[i][j])
            locations.append(face_pb2.Location(Loc=lc))
        try:
            yield face_pb2.LocationsStream(
                ID=response.ID,
                Face_locations=locations,
                Face_names=face_names,
            )
        except Exception as ex:
            traceback.print_exc()


# def yieldLocations(stub, worker_id):
#     for location in send_message(stub, worker_id):
#         try:
#             yield location
#         except StopIteration as si:
#             print("StopIteration")
#             traceback.print_exc()
#         except Exception as ex:
#             traceback.print_exc()

def run():
    p = []
    worker_num = 1
    if 'PODNAME' in os.environ:
        name = os.environ['PODNAME']
    else:
        name = 'default'
    if 'NODE_HOST' in os.environ:
        global _HOST
        _HOST = os.environ['NODE_HOST']
    if 'DAEMON_PORT' in os.environ:
        global _PORT
        _PORT = os.environ['DAEMON_PORT']

    for worker_id in range(0, worker_num):
        worker_name = name + '_' + str(worker_id)
        print(worker_name + ' start')
        p.append(Process(target=client, args=(worker_name, worker_num,)))
        p[worker_id].start()

    worker_id = worker_num + 1
    worker_name = name + '_' + str(worker_id)
    print(worker_name + ' start')
    p.append(Process(target=httpServer(), args=(worker_name, worker_num,)))
    p[worker_id].start()

def client(worker_id, worker_num):
    with grpc.insecure_channel(_HOST + ':' + _PORT) as channel:
        stub = face_pb2_grpc.FaceServiceStub(channel)
        try:
            stub.DisplayLocations(send_message(stub, worker_id))
        except StopIteration as si:
            print("StopIteration")
            traceback.print_exc()
        except Exception as ex:
            traceback.print_exc()



# Create arrays of known face encodings and their names
# faceLock = threading.Lock()
known_face_encodings = []
known_face_names = []

class HTTPRequestHandler(server.SimpleHTTPRequestHandler):
    def log_request(self, format, *args):
        return

    def do_POST(self):
        filename = Path(os.path.basename(self.path))
        file_length = int(self.headers['Content-Length'])
        if str(filename) == 'frame.jpg':
            # result = detect(self.rfile.read(file_length)).encode('utf-8')
            result = ''
            self.send_response(404, 'NotFound')
            self.end_headers()
            self.wfile.write(result)
        else:
            output = PurePosixPath('/tmp').joinpath(filename.name)
            with open(str(output), 'wb') as output_file:
                output_file.write(self.rfile.read(file_length))
            imageEncoding = face_recognition.face_encodings(face_recognition.load_image_file(str(output)))[0]
            # with faceLock:
            known_face_encodings.append(imageEncoding)
            known_face_names.append(str(filename.with_suffix('')))
            self.send_response(201, 'Created')
            self.end_headers()

class MTHTTPServer(ThreadingMixIn, HTTPServer):
    pass

def httpServer():
    server = MTHTTPServer(("", 80), HTTPRequestHandler)
    server.serve_forever()

if __name__ == '__main__':
    run()
