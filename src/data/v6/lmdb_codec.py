import pickle
import zlib


_COMPRESSED_MAGIC = b"LMZ1"


def encode_lmdb_payload(data):
    raw = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    compressed = zlib.compress(raw, level=1)
    if len(compressed) + len(_COMPRESSED_MAGIC) < len(raw):
        return _COMPRESSED_MAGIC + compressed
    return raw


def decode_lmdb_payload(val_bytes):
    if not isinstance(val_bytes, (bytes, bytearray)):
        val_bytes = bytes(val_bytes)
    if val_bytes.startswith(_COMPRESSED_MAGIC):
        val_bytes = zlib.decompress(val_bytes[len(_COMPRESSED_MAGIC):])
    return pickle.loads(val_bytes)


# Backward-compatible aliases for any in-flight imports.
encode_lmdb_value = encode_lmdb_payload
decode_lmdb_value = decode_lmdb_payload
