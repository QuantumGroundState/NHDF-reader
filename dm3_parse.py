"""
DM3 Binary Parser - No external DM libraries required.
Walks the raw binary structure of a DM3 file and prints all tag groups,
tag entries, and tag data in full detail.
"""

import struct
import sys
import os

FILE_PATH = r"C:\Users\MatthewMecklenburg\Desktop\CoWork_5511\t_converter_data\550C_Nanoport_Si_Al_Aligned.dm3"

# DM3 data type codes
DM3_DATATYPES = {
    2:  ('int16',   'h', 2),
    3:  ('int32',   'i', 4),
    4:  ('uint16',  'H', 2),
    5:  ('uint32',  'I', 4),
    6:  ('float32', 'f', 4),
    7:  ('float64', 'd', 8),
    8:  ('bool',    'B', 1),
    9:  ('int8',    'b', 1),
    10: ('uint8',   'B', 1),
    11: ('int64',   'q', 8),
    12: ('uint64',  'Q', 8),
}

# DM3 tag type markers
TAG_GROUP  = 20  # 0x14
TAG_ENTRY  = 21  # 0x15

# Encoding type IDs in tag data
ENCODE_SIMPLE  = 2
ENCODE_STRING  = 18
ENCODE_ARRAY   = 20
ENCODE_STRUCT  = 15
ENCODE_UNKNOWN = 0

MAX_PRINT_DEPTH = 20      # limit recursion depth for printing
MAX_ARRAY_PRINT = 8       # print at most N elements of large arrays

class DM3Parser:
    def __init__(self, filepath):
        with open(filepath, 'rb') as f:
            self.data = f.read()
        self.pos = 0
        self.file_size = len(self.data)
        self.byte_order = '>'   # big-endian by default; updated after header
        self.endian_char = '>'

        # Tracking special sections
        self.image_data_offset = None
        self.image_data_bytes  = None
        self.found_tags = {}    # path -> value

    # ------------------------------------------------------------------ #
    #  Low-level readers                                                   #
    # ------------------------------------------------------------------ #
    def read_bytes(self, n):
        chunk = self.data[self.pos:self.pos+n]
        if len(chunk) < n:
            raise EOFError(f"Unexpected EOF at offset {self.pos}: wanted {n}, got {len(chunk)}")
        self.pos += n
        return chunk

    def read_uint8(self):
        v, = struct.unpack_from('B', self.data, self.pos)
        self.pos += 1
        return v

    def read_int32_be(self):
        v, = struct.unpack_from('>i', self.data, self.pos)
        self.pos += 4
        return v

    def read_uint32_be(self):
        v, = struct.unpack_from('>I', self.data, self.pos)
        self.pos += 4
        return v

    def read_int32(self):
        fmt = self.endian_char + 'i'
        v, = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += 4
        return v

    def read_uint32(self):
        fmt = self.endian_char + 'I'
        v, = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += 4
        return v

    def read_uint64(self):
        fmt = self.endian_char + 'Q'
        v, = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += 8
        return v

    def read_int16_be(self):
        v, = struct.unpack_from('>h', self.data, self.pos)
        self.pos += 2
        return v

    # ------------------------------------------------------------------ #
    #  Header                                                              #
    # ------------------------------------------------------------------ #
    def parse_header(self):
        print("=" * 70)
        print("FILE HEADER (first 12 bytes)")
        print("=" * 70)

        header_start = self.pos
        version   = self.read_int32_be()
        file_size = self.read_int32_be()   # stored as int32 in DM3 (4 bytes)
        # byte_order: 0 = big-endian, 1 = little-endian
        byte_order_flag = self.read_int32_be()

        print(f"  Offset 0x{header_start:08X} | version      = {version}")
        print(f"  Offset 0x{header_start+4:08X} | file_size    = {file_size} bytes  ({file_size/1024/1024:.2f} MB)")
        print(f"  Offset 0x{header_start+8:08X} | byte_order   = {byte_order_flag}  ({'little-endian' if byte_order_flag else 'big-endian'})")
        print(f"  (actual file size on disk = {self.file_size} bytes = {self.file_size/1024/1024:.2f} MB)")
        print()

        if byte_order_flag == 1:
            self.endian_char = '<'
        else:
            self.endian_char = '>'

        return version, file_size, byte_order_flag

    # ------------------------------------------------------------------ #
    #  Tag tree                                                            #
    # ------------------------------------------------------------------ #
    def read_tag_name(self):
        """Read a 2-byte big-endian length followed by that many bytes (Latin-1)."""
        length = self.read_int16_be()
        if length < 0 or length > 4096:
            # Sanity guard
            raise ValueError(f"Tag name length {length} at offset {self.pos-2} seems wrong")
        if length == 0:
            return ""
        raw = self.read_bytes(length)
        return raw.decode('latin-1', errors='replace')

    def parse_tag_group(self, depth=0, path="root"):
        """Parse a TagGroup and return number of entries."""
        indent = "  " * depth

        # is_open (1 byte), is_sorted (1 byte), num_entries (4 bytes big-endian)
        is_open   = self.read_uint8()
        is_sorted = self.read_uint8()
        num_entries = self.read_uint32_be()

        if depth <= MAX_PRINT_DEPTH:
            print(f"{indent}TagGroup '{path}': open={is_open}, sorted={is_sorted}, entries={num_entries}")

        for i in range(num_entries):
            self.parse_tag_entry(depth=depth+1, path=path, index=i)

    def parse_tag_entry(self, depth=0, path="root", index=0):
        """Parse a single tag entry (either sub-group or data)."""
        indent = "  " * depth

        tag_type = self.read_uint8()   # 20 = TagGroup, 21 = TagData
        tag_name = self.read_tag_name()

        entry_path = f"{path}.{tag_name}" if tag_name else f"{path}[{index}]"

        if tag_type == TAG_GROUP:
            # recurse
            self.parse_tag_group(depth=depth, path=entry_path)
        elif tag_type == TAG_ENTRY:
            # Tag data follows
            self.parse_tag_data(depth=depth, path=entry_path)
        else:
            raise ValueError(f"Unknown tag type {tag_type} at offset 0x{self.pos-1:X} (path={entry_path})")

    def parse_tag_data(self, depth=0, path=""):
        """Parse tag data: delimiter, encoding info, then value."""
        indent = "  " * depth

        # 4-byte magic: '%%%%' (0x25252525)
        magic = self.read_bytes(4)
        if magic != b'%%%%':
            raise ValueError(f"Expected '%%%%' at 0x{self.pos-4:X}, got {magic!r} (path={path})")

        # number of type descriptors (big-endian int32)
        num_descriptors = self.read_int32_be()

        # Read type descriptors (each is big-endian int32)
        type_descriptors = []
        for _ in range(num_descriptors):
            type_descriptors.append(self.read_int32_be())

        if not type_descriptors:
            if depth <= MAX_PRINT_DEPTH:
                print(f"{indent}TagData '{path}': (no descriptors)")
            return

        # Decode and read the value
        try:
            value, enc_summary = self.decode_tag_value(type_descriptors, path)
        except Exception as e:
            value = f"<ERROR: {e}>"
            enc_summary = f"descriptors={type_descriptors}"

        # Store for later analysis
        self.found_tags[path] = value

        if depth <= MAX_PRINT_DEPTH:
            print(f"{indent}TagData  '{path}': [{enc_summary}] = {self._repr(value)}")

    def _repr(self, value):
        if isinstance(value, (list, tuple)) and len(value) > MAX_ARRAY_PRINT:
            return f"[{', '.join(repr(v) for v in value[:MAX_ARRAY_PRINT])}, ... ({len(value)} elements total)]"
        if isinstance(value, bytes) and len(value) > 32:
            return f"<bytes len={len(value)}, first 16: {value[:16].hex()}>"
        return repr(value)

    def decode_tag_value(self, descriptors, path):
        """Decode a tag value given its type descriptor list."""
        enc_type = descriptors[0]

        if enc_type == ENCODE_STRUCT:  # 15 - struct
            return self.decode_struct(descriptors, path)

        elif enc_type == ENCODE_STRING:  # 18 - string
            str_len = self.read_uint32_be()
            raw = self.read_bytes(str_len * 2)  # UTF-16 BE
            value = raw.decode('utf-16-be', errors='replace')
            return value, "string(utf-16-be)"

        elif enc_type == ENCODE_ARRAY:  # 20 - array
            return self.decode_array(descriptors, path)

        elif enc_type in DM3_DATATYPES:
            label, fmt_char, size = DM3_DATATYPES[enc_type]
            fmt = self.endian_char + fmt_char
            raw = self.read_bytes(size)
            value, = struct.unpack(fmt, raw)
            # Bool: map to True/False
            if enc_type == 8:
                value = bool(value)
            return value, f"simple({label})"

        else:
            return f"<unknown enc_type={enc_type}>", f"unknown({enc_type})"

    def decode_struct(self, descriptors, path):
        """Decode a struct. descriptors = [15, 0, num_fields, type1, type2, ...]"""
        # descriptors[0] = 15 (struct marker)
        # descriptors[1] = 0 (unused / struct name length, always 0 in DM3)
        # descriptors[2] = num fields
        # then pairs: (field_name_len=0, field_type) for each field
        if len(descriptors) < 3:
            return "<struct: too few descriptors>", "struct"
        num_fields = descriptors[2]
        # Field descriptors start at index 3, each field = 2 ints: (name_len, type)
        fields = []
        idx = 3
        for fi in range(num_fields):
            if idx + 1 >= len(descriptors):
                break
            field_name_len = descriptors[idx]     # always 0 in DM3
            field_type     = descriptors[idx+1]
            idx += 2
            fields.append(field_type)

        values = []
        for ftype in fields:
            if ftype in DM3_DATATYPES:
                label, fmt_char, size = DM3_DATATYPES[ftype]
                fmt = self.endian_char + fmt_char
                raw = self.read_bytes(size)
                v, = struct.unpack(fmt, raw)
                values.append(v)
            else:
                values.append(f"<unknown field type {ftype}>")
        return tuple(values), f"struct(fields={fields})"

    def decode_array(self, descriptors, path):
        """Decode an array. descriptors = [20, element_type, ..., array_length]"""
        # descriptors[0] = 20 (array marker)
        # descriptors[1] = element encoding type
        # If element_type == 15 (struct), then: [20, 15, 0, num_fields, t1,t2,...,tN, array_len]
        # If element_type == 18 (string), there may be extra info
        # Last descriptor is always the array length
        if len(descriptors) < 3:
            return [], "array(empty)"

        elem_type = descriptors[1]
        array_len = descriptors[-1]   # last descriptor = length

        if elem_type == ENCODE_STRUCT:
            # struct array: [20, 15, 0, num_fields, t1..tN, array_len]
            # reconstruct struct descriptor from descriptors[1..-1]
            struct_desc = descriptors[1:-1]
            # Each struct element: parse using struct_desc
            if len(struct_desc) < 3:
                return [], f"array-of-struct(len={array_len})"
            num_fields = struct_desc[2]
            field_types = []
            idx = 3
            for _ in range(num_fields):
                if idx + 1 < len(struct_desc):
                    field_types.append(struct_desc[idx+1])
                    idx += 2
            # Compute struct element size
            elem_size = sum(DM3_DATATYPES[ft][2] for ft in field_types if ft in DM3_DATATYPES)

            total_bytes = elem_size * array_len
            # For large arrays (e.g. image data), just note offset and skip
            if total_bytes > 65536:
                offset_start = self.pos
                # Special detection: if path contains 'Data' or 'data', this is image data
                if 'Data' in path.split('.')[-1] or path.endswith('.Data'):
                    self.image_data_offset = offset_start
                    self.image_data_bytes  = total_bytes
                    print(f"{'  '*10}*** IMAGE DATA ARRAY FOUND ***")
                    print(f"{'  '*10}    Path            : {path}")
                    print(f"{'  '*10}    Byte offset     : 0x{offset_start:X} ({offset_start})")
                    print(f"{'  '*10}    Total bytes     : {total_bytes}")
                    print(f"{'  '*10}    Elem type       : struct(fields={field_types})")
                    print(f"{'  '*10}    Array length    : {array_len}")
                self.pos += total_bytes
                return f"<array-of-struct len={array_len}, {total_bytes} bytes at 0x{offset_start:X}>", \
                       f"array-of-struct(len={array_len},fields={field_types})"

            values = []
            for _ in range(array_len):
                row = []
                for ft in field_types:
                    if ft in DM3_DATATYPES:
                        label, fmt_char, size = DM3_DATATYPES[ft]
                        fmt = self.endian_char + fmt_char
                        raw = self.read_bytes(size)
                        v, = struct.unpack(fmt, raw)
                        row.append(v)
                if len(row) == 1:
                    values.append(row[0])
                else:
                    values.append(tuple(row))
            return values, f"array-of-struct(len={array_len},fields={field_types})"

        elif elem_type == ENCODE_STRING:
            # array of strings - rare, treat as single string
            str_len = array_len  # array_len is # of chars
            raw = self.read_bytes(str_len * 2)
            value = raw.decode('utf-16-be', errors='replace')
            return value, f"array-of-string(len={array_len})"

        elif elem_type in DM3_DATATYPES:
            label, fmt_char, size = DM3_DATATYPES[elem_type]
            total_bytes = size * array_len

            # Detect large data arrays (image data)
            if total_bytes > 65536:
                offset_start = self.pos
                tag_leaf = path.split('.')[-1]
                if tag_leaf in ('Data', 'data') or 'Data' in tag_leaf:
                    self.image_data_offset = offset_start
                    self.image_data_bytes  = total_bytes
                    print(f"{'  '*10}*** IMAGE DATA ARRAY FOUND ***")
                    print(f"{'  '*10}    Path            : {path}")
                    print(f"{'  '*10}    Byte offset     : 0x{offset_start:X} ({offset_start})")
                    print(f"{'  '*10}    Total bytes     : {total_bytes}")
                    print(f"{'  '*10}    Element type    : {label} ({size} bytes each)")
                    print(f"{'  '*10}    Array length    : {array_len}")
                self.pos += total_bytes
                return f"<array len={array_len}, dtype={label}, {total_bytes} bytes at 0x{offset_start:X}>", \
                       f"array({label}, len={array_len})"

            fmt = self.endian_char + fmt_char * array_len
            raw = self.read_bytes(total_bytes)
            values = list(struct.unpack(fmt, raw))
            return values, f"array({label}, len={array_len})"

        else:
            return f"<unknown array elem_type={elem_type}>", f"array(unknown={elem_type})"

    # ------------------------------------------------------------------ #
    #  Post-parse analysis                                                 #
    # ------------------------------------------------------------------ #
    def summarize(self):
        print()
        print("=" * 70)
        print("POST-PARSE SUMMARY")
        print("=" * 70)

        # ---- ImageData ----
        print()
        print("--- ImageData section ---")
        idata_keys = [k for k in self.found_tags if 'ImageData' in k or 'imagedata' in k.lower()]

        # Data type
        dt_key = next((k for k in self.found_tags if k.endswith('.DataType') and 'ImageData' in k), None)
        if dt_key:
            dt_val = self.found_tags[dt_key]
            print(f"  DataType field: {dt_val}  (path: {dt_key})")
            dm3_dtype_map = {
                1: 'int16', 2: 'float32', 3: 'complex64',
                4: '???',   5: '???',     6: 'uint8',
                7: 'int32', 8: '???',     9: 'int8',
                10: 'uint16', 11: 'uint32', 12: 'float64',
                13: 'complex128', 14: 'bool8', 23: 'rgb(uint8x3)'
            }
            print(f"         -> meaning: {dm3_dtype_map.get(dt_val, 'unknown')}")
        else:
            print("  DataType: NOT FOUND")

        # PixelDepth
        pd_key = next((k for k in self.found_tags if k.endswith('.PixelDepth') and 'ImageData' in k), None)
        if pd_key:
            print(f"  PixelDepth: {self.found_tags[pd_key]}  (path: {pd_key})")
        else:
            print("  PixelDepth: NOT FOUND")

        # Dimensions
        print()
        print("  Dimensions:")
        for dim_idx in ['1', '2', '3', '0']:
            dim_key = next((k for k in self.found_tags
                            if f'Dimensions.{dim_idx}' in k and 'ImageData' in k), None)
            if dim_key:
                print(f"    Dimension[{dim_idx}] = {self.found_tags[dim_key]}  (path: {dim_key})")

        # Calibrations
        print()
        print("  Calibrations:")
        for cal_part in ['Brightness', 'Dimension']:
            for sub in ['Origin', 'Scale', 'Units']:
                keys = [k for k in self.found_tags if cal_part in k and k.endswith(f'.{sub}') and 'Calibrations' in k]
                for k in keys:
                    print(f"    {k.split('ImageData.')[-1]} = {self.found_tags[k]!r}")

        # Data array
        print()
        print("  Image Data Array:")
        if self.image_data_offset is not None:
            print(f"    Byte offset  : 0x{self.image_data_offset:X}  ({self.image_data_offset})")
            print(f"    Total bytes  : {self.image_data_bytes}")
        else:
            # Try to find from found_tags
            data_keys = [k for k in self.found_tags if k.endswith('.Data') and 'ImageData' in k]
            for dk in data_keys:
                print(f"    {dk}: {self.found_tags[dk]!r}")

        # Shape analysis
        print()
        print("--- Shape / Image Type Analysis ---")
        dims = {}
        for dim_idx in ['1', '2', '3']:
            dim_key = next((k for k in self.found_tags
                            if f'Dimensions.{dim_idx}' in k and 'ImageData' in k), None)
            if dim_key:
                dims[int(dim_idx)] = self.found_tags[dim_key]

        # Also check index 0 (sometimes used)
        dim0_key = next((k for k in self.found_tags
                         if f'Dimensions.0' in k and 'ImageData' in k), None)
        if dim0_key:
            dims[0] = self.found_tags[dim0_key]

        if dims:
            shape = tuple(dims[k] for k in sorted(dims.keys()))
            print(f"  Dimension dict: {dims}")
            print(f"  Shape (sorted by index): {shape}")
            ndim = len(dims)
            if ndim == 3:
                print("  => This is a 3D SPECTRUM IMAGE (SI)")
                print(f"     Interpretation: (energy_channels x scan_y x scan_x) or similar")
            elif ndim == 2:
                print("  => This is a 2D IMAGE")
            else:
                print(f"  => {ndim}D data")
        else:
            print("  Could not determine shape from tag data.")

        # ImageTags / Meta Data
        print()
        print("--- ImageTags / Meta Data ---")
        meta_keys = [k for k in self.found_tags if 'ImageTags' in k or 'Meta Data' in k or 'MetaData' in k]
        if meta_keys:
            for mk in sorted(meta_keys)[:80]:
                print(f"  {mk} = {self._repr(self.found_tags[mk])}")
            if len(meta_keys) > 80:
                print(f"  ... ({len(meta_keys) - 80} more ImageTags entries)")
        else:
            print("  No ImageTags/MetaData keys found.")

        # EELS-related tags
        print()
        print("--- EELS / Spectrum-related tags ---")
        eels_keys = [k for k in self.found_tags
                     if any(kw in k for kw in ['EELS', 'eels', 'Spectrum', 'spectrum',
                                                'Acquisition', 'acquisition',
                                                'Signal', 'signal', 'EDS', 'EDX'])]
        if eels_keys:
            for ek in sorted(eels_keys)[:60]:
                print(f"  {ek} = {self._repr(self.found_tags[ek])}")
        else:
            print("  No EELS/Spectrum tags found.")

        print()
        print("--- All found tags (full list) ---")
        for k in sorted(self.found_tags.keys()):
            print(f"  {k} = {self._repr(self.found_tags[k])}")

    def run(self):
        print(f"DM3 Parser")
        print(f"File: {FILE_PATH}")
        print(f"File size on disk: {self.file_size} bytes ({self.file_size/1024/1024:.2f} MB)")
        print()

        version, file_size, byte_order_flag = self.parse_header()

        print("=" * 70)
        print("TAG TREE")
        print("=" * 70)
        print()

        # The root of the DM3 tag tree is a TagGroup immediately after the header
        self.parse_tag_group(depth=0, path="root")

        print()
        print(f"Parser finished at offset: 0x{self.pos:X} ({self.pos})")
        print(f"Remaining bytes: {self.file_size - self.pos}")

        self.summarize()


if __name__ == '__main__':
    parser = DM3Parser(FILE_PATH)
    try:
        parser.run()
    except Exception as e:
        import traceback
        print(f"\n*** PARSER ERROR at offset 0x{parser.pos:X}: {e} ***")
        traceback.print_exc()
        print(f"\nPartial results up to error:")
        parser.summarize()
