# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: face.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='face.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\nface.proto\"e\n\x0b\x46rameStream\x12\n\n\x02ID\x18\x01 \x01(\t\x12\x17\n\x0fRgb_small_frame\x18\x02 \x01(\x0c\x12\"\n\x06Status\x18\x03 \x01(\x0e\x32\x12.EnumFaceSvcStatus\x12\r\n\x05\x45rror\x18\x04 \x01(\t\"\x1a\n\x0c\x46rameRequest\x12\n\n\x02ID\x18\x01 \x01(\t\"\x17\n\x08Location\x12\x0b\n\x03Loc\x18\x01 \x03(\x05\"m\n\x0fLocationsStream\x12\n\n\x02ID\x18\x01 \x01(\t\x12!\n\x0e\x46\x61\x63\x65_locations\x18\x02 \x03(\x0b\x32\t.Location\x12\x12\n\nFace_names\x18\x03 \x03(\t\x12\x17\n\x0fRgb_small_frame\x18\x04 \x01(\x0c\"E\n\x10LocationResponse\x12\"\n\x06Status\x18\x01 \x01(\x0e\x32\x12.EnumFaceSvcStatus\x12\r\n\x05\x45rror\x18\x02 \x01(\t*s\n\x11\x45numFaceSvcStatus\x12\x0b\n\x07INVALID\x10\x00\x12\r\n\tSTATUS_OK\x10\x01\x12\x16\n\x12NO_ENOUGH_ARGUMENT\x10\x02\x12\x13\n\x0fPARTIAL_FAILURE\x10\x03\x12\x15\n\x11PERMISSION_DENIED\x10\x04\x32}\n\x0b\x46\x61\x63\x65Service\x12\x31\n\x0eGetFrameStream\x12\r.FrameRequest\x1a\x0c.FrameStream\"\x00\x30\x01\x12;\n\x10\x44isplayLocations\x12\x10.LocationsStream\x1a\x11.LocationResponse\"\x00(\x01\x62\x06proto3'
)

_ENUMFACESVCSTATUS = _descriptor.EnumDescriptor(
  name='EnumFaceSvcStatus',
  full_name='EnumFaceSvcStatus',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='INVALID', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='STATUS_OK', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='NO_ENOUGH_ARGUMENT', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='PARTIAL_FAILURE', index=3, number=3,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='PERMISSION_DENIED', index=4, number=4,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=352,
  serialized_end=467,
)
_sym_db.RegisterEnumDescriptor(_ENUMFACESVCSTATUS)

EnumFaceSvcStatus = enum_type_wrapper.EnumTypeWrapper(_ENUMFACESVCSTATUS)
INVALID = 0
STATUS_OK = 1
NO_ENOUGH_ARGUMENT = 2
PARTIAL_FAILURE = 3
PERMISSION_DENIED = 4



_FRAMESTREAM = _descriptor.Descriptor(
  name='FrameStream',
  full_name='FrameStream',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='ID', full_name='FrameStream.ID', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='Rgb_small_frame', full_name='FrameStream.Rgb_small_frame', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='Status', full_name='FrameStream.Status', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='Error', full_name='FrameStream.Error', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=14,
  serialized_end=115,
)


_FRAMEREQUEST = _descriptor.Descriptor(
  name='FrameRequest',
  full_name='FrameRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='ID', full_name='FrameRequest.ID', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=117,
  serialized_end=143,
)


_LOCATION = _descriptor.Descriptor(
  name='Location',
  full_name='Location',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='Loc', full_name='Location.Loc', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=145,
  serialized_end=168,
)


_LOCATIONSSTREAM = _descriptor.Descriptor(
  name='LocationsStream',
  full_name='LocationsStream',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='ID', full_name='LocationsStream.ID', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='Face_locations', full_name='LocationsStream.Face_locations', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='Face_names', full_name='LocationsStream.Face_names', index=2,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='Rgb_small_frame', full_name='LocationsStream.Rgb_small_frame', index=3,
      number=4, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=170,
  serialized_end=279,
)


_LOCATIONRESPONSE = _descriptor.Descriptor(
  name='LocationResponse',
  full_name='LocationResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='Status', full_name='LocationResponse.Status', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='Error', full_name='LocationResponse.Error', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=281,
  serialized_end=350,
)

_FRAMESTREAM.fields_by_name['Status'].enum_type = _ENUMFACESVCSTATUS
_LOCATIONSSTREAM.fields_by_name['Face_locations'].message_type = _LOCATION
_LOCATIONRESPONSE.fields_by_name['Status'].enum_type = _ENUMFACESVCSTATUS
DESCRIPTOR.message_types_by_name['FrameStream'] = _FRAMESTREAM
DESCRIPTOR.message_types_by_name['FrameRequest'] = _FRAMEREQUEST
DESCRIPTOR.message_types_by_name['Location'] = _LOCATION
DESCRIPTOR.message_types_by_name['LocationsStream'] = _LOCATIONSSTREAM
DESCRIPTOR.message_types_by_name['LocationResponse'] = _LOCATIONRESPONSE
DESCRIPTOR.enum_types_by_name['EnumFaceSvcStatus'] = _ENUMFACESVCSTATUS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

FrameStream = _reflection.GeneratedProtocolMessageType('FrameStream', (_message.Message,), {
  'DESCRIPTOR' : _FRAMESTREAM,
  '__module__' : 'face_pb2'
  # @@protoc_insertion_point(class_scope:FrameStream)
  })
_sym_db.RegisterMessage(FrameStream)

FrameRequest = _reflection.GeneratedProtocolMessageType('FrameRequest', (_message.Message,), {
  'DESCRIPTOR' : _FRAMEREQUEST,
  '__module__' : 'face_pb2'
  # @@protoc_insertion_point(class_scope:FrameRequest)
  })
_sym_db.RegisterMessage(FrameRequest)

Location = _reflection.GeneratedProtocolMessageType('Location', (_message.Message,), {
  'DESCRIPTOR' : _LOCATION,
  '__module__' : 'face_pb2'
  # @@protoc_insertion_point(class_scope:Location)
  })
_sym_db.RegisterMessage(Location)

LocationsStream = _reflection.GeneratedProtocolMessageType('LocationsStream', (_message.Message,), {
  'DESCRIPTOR' : _LOCATIONSSTREAM,
  '__module__' : 'face_pb2'
  # @@protoc_insertion_point(class_scope:LocationsStream)
  })
_sym_db.RegisterMessage(LocationsStream)

LocationResponse = _reflection.GeneratedProtocolMessageType('LocationResponse', (_message.Message,), {
  'DESCRIPTOR' : _LOCATIONRESPONSE,
  '__module__' : 'face_pb2'
  # @@protoc_insertion_point(class_scope:LocationResponse)
  })
_sym_db.RegisterMessage(LocationResponse)



_FACESERVICE = _descriptor.ServiceDescriptor(
  name='FaceService',
  full_name='FaceService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=469,
  serialized_end=594,
  methods=[
  _descriptor.MethodDescriptor(
    name='GetFrameStream',
    full_name='FaceService.GetFrameStream',
    index=0,
    containing_service=None,
    input_type=_FRAMEREQUEST,
    output_type=_FRAMESTREAM,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='DisplayLocations',
    full_name='FaceService.DisplayLocations',
    index=1,
    containing_service=None,
    input_type=_LOCATIONSSTREAM,
    output_type=_LOCATIONRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_FACESERVICE)

DESCRIPTOR.services_by_name['FaceService'] = _FACESERVICE

# @@protoc_insertion_point(module_scope)