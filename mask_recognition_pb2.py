# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mask_recognition.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x16mask_recognition.proto\"L\n\x08\x42\x36\x34Image\x12\x10\n\x08\x62\x36\x34image\x18\x01 \x01(\t\x12\r\n\x05width\x18\x02 \x01(\x05\x12\x0e\n\x06height\x18\x03 \x01(\x05\x12\x0f\n\x07\x63hannel\x18\x04 \x01(\x05\"\\\n\nPrediction\x12\x0e\n\x06status\x18\x01 \x01(\t\x12\x13\n\x0b\x62\x36\x34\x66\x61\x63\x65locs\x18\x02 \x01(\t\x12\x13\n\x0b\x62\x36\x34noselocs\x18\x03 \x01(\t\x12\x14\n\x0c\x62\x36\x34nosepreds\x18\x04 \x01(\t2H\n\x0fMaskRecognition\x12\x35\n\x19make_inference_from_frame\x12\t.B64Image\x1a\x0b.Prediction\"\x00\x62\x06proto3')



_B64IMAGE = DESCRIPTOR.message_types_by_name['B64Image']
_PREDICTION = DESCRIPTOR.message_types_by_name['Prediction']
B64Image = _reflection.GeneratedProtocolMessageType('B64Image', (_message.Message,), {
  'DESCRIPTOR' : _B64IMAGE,
  '__module__' : 'mask_recognition_pb2'
  # @@protoc_insertion_point(class_scope:B64Image)
  })
_sym_db.RegisterMessage(B64Image)

Prediction = _reflection.GeneratedProtocolMessageType('Prediction', (_message.Message,), {
  'DESCRIPTOR' : _PREDICTION,
  '__module__' : 'mask_recognition_pb2'
  # @@protoc_insertion_point(class_scope:Prediction)
  })
_sym_db.RegisterMessage(Prediction)

_MASKRECOGNITION = DESCRIPTOR.services_by_name['MaskRecognition']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _B64IMAGE._serialized_start=26
  _B64IMAGE._serialized_end=102
  _PREDICTION._serialized_start=104
  _PREDICTION._serialized_end=196
  _MASKRECOGNITION._serialized_start=198
  _MASKRECOGNITION._serialized_end=270
# @@protoc_insertion_point(module_scope)