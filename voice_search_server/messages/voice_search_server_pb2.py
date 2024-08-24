# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: messages/voice_search_server.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\"messages/voice_search_server.proto\x12\x0b\x61udiostream\"l\n\x19StreamingRecognizeRequest\x12\x38\n\x10streaming_config\x18\x01 \x01(\x0b\x32\x1e.audiostream.RecognitionConfig\x12\x15\n\raudio_content\x18\x02 \x01(\x0c\"Q\n\x11RecognitionConfig\x12\x19\n\x11sample_rate_hertz\x18\x01 \x01(\x05\x12\r\n\x05model\x18\x02 \x01(\t\x12\x12\n\nrequest_id\x18\x03 \x01(\t\"z\n\x1aStreamingRecognizeResponse\x12\"\n\x05\x65rror\x18\x01 \x01(\x0b\x32\x13.audiostream.Status\x12\x38\n\x07results\x18\x02 \x03(\x0b\x32\'.audiostream.StreamingRecognitionResult\"\'\n\x06Status\x12\x0c\n\x04\x63ode\x18\x01 \x01(\x05\x12\x0f\n\x07message\x18\x02 \x01(\t\"V\n\x11WordAlignmentInfo\x12\x0c\n\x04word\x18\x01 \x01(\t\x12\r\n\x05start\x18\x02 \x01(\x02\x12\x10\n\x08\x64uration\x18\x03 \x01(\x02\x12\x12\n\nconfidence\x18\x04 \x01(\x02\"\xa2\x01\n\x1aStreamingRecognitionResult\x12?\n\x0c\x61lternatives\x18\x01 \x03(\x0b\x32).audiostream.SpeechRecognitionAlternative\x12\x31\n\talignment\x18\x02 \x03(\x0b\x32\x1e.audiostream.WordAlignmentInfo\x12\x10\n\x08is_final\x18\x03 \x01(\x08\"F\n\x1cSpeechRecognitionAlternative\x12\x12\n\ntranscript\x18\x01 \x01(\t\x12\x12\n\nconfidence\x18\x02 \x01(\x02\"5\n\rEnrollRequest\x12\x15\n\raudio_content\x18\x01 \x01(\x0c\x12\r\n\x05label\x18\x02 \x01(\t\"G\n\x0e\x45nrollResponse\x12\x12\n\nrequest_id\x18\x01 \x01(\t\x12\x12\n\ntranscript\x18\x02 \x01(\t\x12\r\n\x05label\x18\x03 \x01(\t\"\x10\n\x0eRetrainRequest\"\"\n\x0fRetrainResponse\x12\x0f\n\x07message\x18\x01 \x01(\t2\x82\x02\n\x06Speech\x12k\n\x12StreamingRecognize\x12&.audiostream.StreamingRecognizeRequest\x1a\'.audiostream.StreamingRecognizeResponse\"\x00(\x01\x30\x01\x12\x43\n\x06\x45nroll\x12\x1a.audiostream.EnrollRequest\x1a\x1b.audiostream.EnrollResponse\"\x00\x12\x46\n\x07Retrain\x12\x1b.audiostream.RetrainRequest\x1a\x1c.audiostream.RetrainResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'messages.voice_search_server_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_STREAMINGRECOGNIZEREQUEST']._serialized_start=51
  _globals['_STREAMINGRECOGNIZEREQUEST']._serialized_end=159
  _globals['_RECOGNITIONCONFIG']._serialized_start=161
  _globals['_RECOGNITIONCONFIG']._serialized_end=242
  _globals['_STREAMINGRECOGNIZERESPONSE']._serialized_start=244
  _globals['_STREAMINGRECOGNIZERESPONSE']._serialized_end=366
  _globals['_STATUS']._serialized_start=368
  _globals['_STATUS']._serialized_end=407
  _globals['_WORDALIGNMENTINFO']._serialized_start=409
  _globals['_WORDALIGNMENTINFO']._serialized_end=495
  _globals['_STREAMINGRECOGNITIONRESULT']._serialized_start=498
  _globals['_STREAMINGRECOGNITIONRESULT']._serialized_end=660
  _globals['_SPEECHRECOGNITIONALTERNATIVE']._serialized_start=662
  _globals['_SPEECHRECOGNITIONALTERNATIVE']._serialized_end=732
  _globals['_ENROLLREQUEST']._serialized_start=734
  _globals['_ENROLLREQUEST']._serialized_end=787
  _globals['_ENROLLRESPONSE']._serialized_start=789
  _globals['_ENROLLRESPONSE']._serialized_end=860
  _globals['_RETRAINREQUEST']._serialized_start=862
  _globals['_RETRAINREQUEST']._serialized_end=878
  _globals['_RETRAINRESPONSE']._serialized_start=880
  _globals['_RETRAINRESPONSE']._serialized_end=914
  _globals['_SPEECH']._serialized_start=917
  _globals['_SPEECH']._serialized_end=1175
# @@protoc_insertion_point(module_scope)
