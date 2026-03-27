#!/usr/bin/env python3
"""
NHDF Spectrum Image Converter  v0.2

Converts Nion Swift .nhdf files to:
  · Raw binary float32  (.dat)  — C-order (ne, ny, nx), same byte layout as
                                   the DM3 data array, no header, no metadata.
                                   A matching .txt sidecar is written alongside.
  · Gatan Digital Micrograph 3  (.dm3)  — opens as EELS Spectrum Image in GMS

DM3 tag structure is copied from  550C_Nanoport_Si_Al_Aligned.dm3
(a real GMS-written EELS SI file).  All 15 ROOT tags are reproduced in the
exact same alphabetical order, including the survey image in ImageList[0] and
the 3-D SI in ImageList[1].

Critical format facts confirmed from reference binary:
  · Strings           → ARRAY of uint16  info=[20,4,N]; N × uint16-LE ASCII
  · Calibration O/S   → float32  (info=[6])
  · Bool flags        → bool8    (info=[8])
  · Color struct      → 3 × int16  info=[15,0,3,0,2,0,2,0,2]
  · Point (2-coord)   → 2 × float32  info=[15,0,2,0,6,0,6]
  · Rectangle         → 4 × float32  info=[15,0,4,0,6,0,6,0,6,0,6]
  · CLUT              → struct-array  info=[20,15,0,3,0,2,0,2,0,2,256]
  · Dimension keys    → 0-based ("0","1","2")
  · DataType 2        = float32   SI data
  · DataType 23       = RGBA int32   survey thumbnail
  · Data axis order   = (ne, ny, nx) C-order
  · Dimensions[0]=nx (fastest), [1]=ny, [2]=ne (slowest)
  · ClassName         = 'ImageSource:Summed'  (NOT 'ImageSlice1D')
  · ImageRef          = 1  (index into ImageList — ImageList[1] is the SI)
  · Summed Dimension  = 2  (collapse ne-axis to produce the 2-D spatial map)
  · Meta Data/Format  = 'Spectrum image'  (lowercase i — must match GMS)
"""

import os, json, struct, datetime, threading, traceback, random
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from io   import BytesIO

import numpy as np
import h5py


# ─────────────────────────────────────────────────────────────────────────────
#  DM3 BINARY WRITER
# ─────────────────────────────────────────────────────────────────────────────

# PageSetup binary blobs copied verbatim from 550C_Nanoport_Si_Al_Aligned.dm3
_WIN32_BYTES = bytes.fromhex(
    '0400000034210000f82a000000000000000000000000000000000000e8030000e8030000e8030000e803000000000000010001000100010001001b10'
)
_WIN32_DEVMODEW_BYTES = bytes.fromhex(
    '420072006f007400680065007200200048004c002d0032003100370030005700200073006500720069006500730000000000000000000000000000000000000001040801dc00c4090fff010001000100ea0a6f08640001000f005802020001005802020000004c006500740074006500720000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100601ff5052495600200000249806004000110101100000180000000000102710271027000010270000000000000000ff000000000100000002000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000b5e705000100050001000000000000000000000102000000000000000000640000000000000000000100010000000000000000000000000000000000000000000000000000000000000000000000000000000100fa0288040000000000000000000001000001000000000100000023070100000001000100000000000000ffff2000070007000700000100000000000081000000000000000000000002000000070000000000000000190000e40c000000000000b80b00007017000000000000dc05000058020000000000000000000000000000feff0000000000000000000000000000000000000000000000000000370c0000da0c000010270000102700001027000010270000a00a0000c2060000b80600002c04000040010000d200000000000a0000000000000000000000000000000000000000000000ffff00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000e4040000530055005200500052004900530045000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000fffffffffefffffffefffffffefffffffefffffffefffffffefffffffefffffffefffffffefffffffefffffffefffffffefffffffefffffffefffffffefffffffefffffffefffffffefffffffefffffffefffffffefffffffffffffffeffffff454e474c495348000000000000000000420052005300500032003000370041002e004500580045000000000000000000420052004c0048004c004100370041002e0044004c004c00000000000000000042005200420033004c004100370041002e0044004c004c00000000000000000042004f00320031003700300057002e0049004e00490000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000090090040000000000000000000000000100ffff000000000e0100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000c60088000000640000000000010100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000006000000f5ffffff00000000000000000000000000000000000000010000000041007200690061006c0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000ffffff00010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000ffffffff6400000000000000000000804d00650063006b006c0065006e0062007500720067000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000100000042524c484c3037412e444c4c00000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000454f5343'
)
_WIN32_DEVNAMESW_BYTES = bytes.fromhex(
    '04001c0034000000420072006f007400680065007200200048004c002d003200310037003000570020007300650072006900650073000000420072006f007400680065007200200048004c002d0032003100370030005700200073006500720069006500730000005500530042003000300031000000'
)

# ── Embedded DM3 structural template ─────────────────────────────────────────
# Compressed (zlib) + base64-encoded binary template derived from a verified
# GMS-created DM3 Spectrum Image file.  The SI-data slot and the file-size /
# data-count fields are zeroed; they are patched at runtime from the NHDF.
# HEAD = first _DM3_TEMPLATE_HEAD_SIZE bytes of the template (everything
#        before the SI data payload including the %%%% descriptor).
# TAIL = remaining bytes (everything after the SI data payload).
_DM3_TEMPLATE_B64 = (
    'eNrt2gd8G1cdB/BnWd6OHadS20AA0ZIWSkLTDNqakdiW7QjsSLXktJRhztKzfMnpTrlhx2kAQ5ll'
    'lVX2KnuEvfemlL33XgXKJhTKCP/3O0k5neSsMj60/28+pzvfu3vv3bt3//dOihCiVfhaWtTn6rgY'
    'GCqXDT2vubplDlueWXDWE0rroaW/cnS0sm49zlpEaWuwTYiYWJO08l5Jmm56do/MuxO64/oFx2hR'
    'G7GYOG3INC0XRY/blleuHSTiou9oWm6pLCuVUme2xnBA/7CW31u0VZVHLMOyK0d0BapdrVYkvD4C'
    'VMjRPCatQrCQSISK6BzTDSOc0KLKHrNseYplHz58zTVjY3GxaqfmHC0+UEIntd7qVEkryqTulA1t'
    'KWXOWWiyO8VFz7CtF+fdU77ibv98UzpOoMh2WrbHRdeIVlYNnjbrqkOlVhKy+oFgW0S7KS06MjGd'
    'C/SZ2IlUp9IH0BGVCLRCFNqgHTqgE7qgG3qgF1ZBH/TDahiANXAaxCAOp8MZcCashdvB7WEd3AHu'
    'CHeCBNwZzoKz4S6wHs6Bc+GucDc4D+4OG2Aj3APOh01wAWyGLbAVtsE94UK4CC6GQbgX3BvuA/eF'
    '7bADhmAYRiAJozAG47ATUnA/uD9MwCTsgjRk4BKYgizkYBp2w6VwGTwALocHwoPgwfAQmIGHggaz'
    'kIcCSJiDIsyDDntgLxhQAhMsKMM+sMEBFzxYgEXYD0twAK6Ag/AweDg8ApbhkfAouBIeDY+Bx8Lj'
    '4PHwBLgKnghPgifDU+CpcDU8DZ4Oz4BnwrPgGng2PAeeC8+D58ML4IXwIngxvAReCtfCy+Dl8Ap4'
    'JbwKXg2vgdfC6+D1cAjeAG+EN8Gb4S3wVngbvB3eAe+Ed8G74T3wXngfvB8+AB+ED8GH4SPwUfgY'
    'fBw+AZ+E6+BTcD18Gj4Dn4XPwefhC/BF+BJ8Gb4CX4WvwdfhG/BN+BZ8G74D34XvwffhB/BD+BH8'
    'GH4CP4Wfwc/hBvgF/BJ+BTfCr+E38Fv4Hfwe/gB/hD/BYfgz3AR/gb/CzfA3+Dv8A/4J1cGoUw0d'
    'u7RSdUxprQwfarrRNS5sIcWScEReaMKgbTUOWaWyIfeHxuS2KGYEvZXUKc0syvrR7UBSlWaZrq05'
    'buPA11tNCmfc4k81kvrcnK3l1RAYOkIN1pRMEx2HEie0WWk4/rgWF6LJZalqJq0hz7Wynr0gl0ID'
    'be+o4+olzZWFSW1/qJ6UfLtgcs7WSxlp52mOpdVfr7XmysG6vPTggN5+7ddv3F6Xl24eL6+2ca1U'
    '0prMGHbSbGJCL+luYwHrd+pIqbZtUhquRgUVi9JuXpZ/kf0pJ1U0aYI1rtuFYc0ONVE3JZsL0qaK'
    'h1I6J6zFhrr4eZ5TTTq5ysREz6Smm1maIMtUIXxf0UMqdaY21EteaYUuproueuVQYY8XTl2mtopn'
    'y5rtSL9PzIzbeiE01WpLIJe1dcft8kqzVH19P3W64LE7cOy6umOnHbnC4Z10SX3+UTmZn9f3eXUF'
    'R5BZ3YQ0PCH3n5IeHJK1PGrNxhai2zZpLUht1pChsntSzpR09ANNknpTTlYa9PrQJI26Sc7WTGdO'
    '2naT5L5KsqE1O7kr5ezWHT2cEBPd/utKTis6mKFGaKI+M6M7M/QyMpO3ykvhYmZmFjVnxkEtG3pk'
    '15Squ1msFdPspar9OGty1fD5g3sphK2mhnJtHYEo6y4ZTdq5c9pUdzCVDCZ1oi/3Vl/JaldHxw/Q'
    'a8ilulmwFjOWo6uMQy3Sh9uaGJbz2oJu2TivPS7WBDvEcV8cT+YaU87lllWShZzl1ytUn4Fp08lr'
    'BqXj7lt2qXqj2tNzc46sPl0dgWpEViqWIltWZXZi5yxv95/WVbt1uVh9OWtoaOrrfsXVzb+lbSJ2'
    'GMmYiKkWGVIvqQvyllz28g2RHad62TF6aNQtr72kR6pv8tFqUpKeNexpo842ohn6rI0XeL+z0Xt7'
    '4O3T36VqT7vqx6cmNQyEyjbq4K7TbGilatRG4voa/rvL+U/kGhenV7pUteVkIXhwNeJEVSuHMqD1'
    'eeLmw4ePqIWI4EJTSMEYY+zWqRr7Of4zxhjHf47/jDF224r/3BqMMcbxnzHGGMd/xhhjHP8ZY4xx'
    '/GeMMcbxnzHGGMd/xhhjHP8ZY4xx/GeMMcbxnzHGGMd/xhhjHP8ZY4xx/GeMMcbxnzHGGMd/xhhj'
    '/5n4f/3112M5dOgQNxBjjHH8Z4wxdiuP/1dffTWWgwcPcgMxxthtKP4zxhjj+M8YY4zjP2OMMY7/'
    'jDHGOP4zxhj7/47/N910k1BLVTQaFevWrcPCGGOM4z9jjLFbR/yPi86k5mq5pbJcT2hvCy1tZ9BH'
    'THQn9ZI0Hd0yHZwQiQsRPGqZPkO71GgRF90Zfb80krLszgcTo8i1K1XSijKnFR21U7TERM/4ZDax'
    'W9qqJH9fXHSM2FJzZaFyfitOpXGJlp4t4h5im9hMn1vExbSorU1UbHSXVpJNTtiQEiWhiaKQIiHS'
    'Yo4+NVEWttCFITbS6ffE5yZaLqCtEh1p0dqmT4v+2ihcOrNEZ2wUWZESM/jboc+Y6Jw29X2eTCX9'
    'UTPUGn+8ZPmc0K6Xew+8LLTr++t+fXZo12Dfuotj/raIVptM3SjsaYuJ3hHN0GdtzVU3Bztb6X4N'
    '23px3jWlU9kVF+1p2qWbgbzbBW5RWzavGbJ+//J22k8X5DpNGlHduVp/EJUSxSmU883PaUPHKCdi'
    'UlP/z3Je7kk2y/lTT1k1eKycpdhNNTo9qTtlQ1uq3htZCJ6gMuqknh1V9zGUSeWSmj6LkabPYmuo'
    'x1wjGp7FnsZdNKs7yceTet+Zo4aksl3NSFziaaarz+l5dLzq89uWdeliq4e3D3mulUpWW7lj1NRm'
    'jdqDjFagOnSkF6RNbRVqnJ6sukhXX9DdYFJrC+oWO1qTjGbTw+5S0EBB7XExMJSn82RitFCUiXHb'
    '8srBDI6QuOirHuPn03hAbMjUjCWXLtBI5GR+Hk93kzseHRWjYoLCQUycUcksUKXEhO64fpCl1h41'
    'C4lRU9rF4BV1qMQnfvSH9Mj1UvPZ7gqH+E9e76Rm763WuqUSi1cNuVZJzydMrzQr7eC1rEXBTVue'
    '7lbGlnO1ALE2ZeYNryATZcOz6aIdurV0CbpZDN2wngltVhqJstT2NnsE4pMUU/dQvLQpukpRqERb'
    'S5gUY5doy6D0WdpnUF5rsvPWYmKf6kyJRd0sWItOqJINQ0Lk1IaEtqy2cDJnUP+flK6WqMVaKrd9'
    'zLJLmtskk74sDQpS5GkosIVH0SVBQ0p1qKETs3qRutOxu093xrbyFLGpxUXleapFqK50Wfoxvkke'
    'A+HCR6hggyowS39rGKNU7g3PSbcpFxNWQ+h74djzPkAlqkSnefhDmnfsENhtGYWVM1eJK2Wu0o6T'
    'eTSnNx3f41vE+XT/tuFzE8bzDXQrNotBWi6mT7UvITJi8hiThLP/O1OCz99xTUto1767r98e2mU8'
    'ZNdMaNee8elzY6If0TlreXZe1oJMrct0UDuOGJrjrHCNa45OhLJ0HR5dT562B+kv1YFKeHCp3yat'
    'RNYr1YXmmIikCqIyOaurWGXoQsWm5FwwqcVPmtCWpE1RsPGsbiQh/jUmrqY6lGQhURv5wqMiBaWU'
    'iXInrYIMjSQqa8tzA6NpsCU6pikw7aXFEov0GRN9k7pZiTeN7UrZ9U/JfZ5uy0LloPrqttHDGtsl'
    '97tJK++pGJ2e3SPzbioZPGwVyl+VofomhuW8tqBbNrLvogtJ2toiBXl6bsLzhS6VlNHKdSFeJQyk'
    'nDEayAspM9OkCVQDppzLLYuaMGddiigbygA3ppD23FDwRRVztmY6cxT5amEwPTfnyOqxanDqr4xP'
    'kcq6PbQOT9KOc87ydn8KunZKOq5lS3/S60+phi3PLFRbpieQT3SFshvqIsSmISFOH4mLeJP86+9U'
    'J+ren6P7Id2mR6jZAo0VqqGy0vXKLZVnvWNc0jBei/mnBWraWVm3hdatJ7pWZdxAf229sxB/Oc/f'
    'VsuZfz9yRC3U2nSbt2wOdfduWu6tmql6Xlg1n+qCHl/7F6OW6EO+M0m5oHrZpY0FxDdMUuTMIz46'
    'tMxRJFQBV8VTE9su7VV7kmKs6ctoS7S1/TsiE2s93y/2l91WZ4HW/eKySITWl0Va6XOCIpSLWGv/'
    'G16AWwLrSGWtbux4Krs1eFwytWv6LHGduEvrhoHtD77i4cfK07+rXbW82W3Pydz762jJTubup7ZX'
    'i3eJK2jOcZHYSk/JJpp5jIkhmmGM0mxkC+3bSEtSXEj/NtIxW4SaRG6k4y6k/RfR0RfSFHAT7d+G'
    'vx5GOU6NZpMTE4JmJLZ01BbGkax+QIqJ0VxudErQO7B6oVJzTJFJT+WmhlI5QQHSMjzso810WW1d'
    'IEYsw7JVAPC36q9jw4AQu7cmJ6vX/oLu8lnqy6AbK4Hi5pZ059afLU5e+ZuOnWd8aNsjv0D7JqrR'
    'rfNoPtW/z6us1d87aNntNxBdv0VxxsNcRUWWaYo3KhpkaF7j0L9FvIKomUr4yDRG+hM7doT2L9Hs'
    'zqWZkUSKimQu7WOMMcYYY4wxxhg7EXHRX/tCVf1S4zT5RlUcjIrbi820cbxvVnN136ye3PewGZEW'
    'U5RDhj7TdG6GtgdRid6sNF3dlMbRn7Jjojs375VmTU03/P+EUPuBJBIX3fg6PWUW5P7Gn3IG/F+s'
    '1BdPM/jPB84xfhKofeF9iLZGcX6f//tFxlL/N6D240uz3wFW/Pb8qS3+D1JfoeUd0aM341+n1IH2'
)
_DM3_TEMPLATE_HEAD_SIZE = 48490   # bytes in template before the SI-data payload


class DM3Writer:
    """
    GMS-compatible DM3 Spectrum Image writer.

    Reproduces the exact 15-tag ROOT structure of 550C_Nanoport_Si_Al_Aligned.dm3.
    ImageList[0] = 2-D RGBA survey thumbnail (energy-summed projection).
    ImageList[1] = 3-D float32 EELS Spectrum Image.
    ImageSourceList[0] points to ImageList[1] via ImageRef=1 and
    ClassName='ImageSource:Summed', Summed Dimension=2.
    """

    def __init__(self):
        self._b = BytesIO()
        self._group_stack = []   # [(n_entries_pos, child_count), ...]

    # ── byte-level primitives ─────────────────────────────────────────────────

    def _u32be(self, v): self._b.write(struct.pack('>I', int(v) & 0xFFFFFFFF))
    def _u16be(self, v): self._b.write(struct.pack('>H', int(v) & 0xFFFF))
    def _byte(self,  v): self._b.write(struct.pack('B',  int(v) & 0xFF))
    def _f32le(self, v): self._b.write(struct.pack('<f', float(v)))
    def _u32le(self, v): self._b.write(struct.pack('<I', int(v) & 0xFFFFFFFF))
    def _i32le(self, v): self._b.write(struct.pack('<i', int(v)))
    def _i16le(self, v): self._b.write(struct.pack('<h', int(v)))
    def _u16le(self, v): self._b.write(struct.pack('<H', int(v) & 0xFFFF))

    # ── tag structural elements ───────────────────────────────────────────────

    def _tg_open(self, sorted_=True, open_=False):
        """Begin a TagGroup.  n_entries is auto-counted and back-patched by _tg_close."""
        self._byte(1 if sorted_ else 0)
        self._byte(1 if open_   else 0)
        pos = self._b.tell()          # position of the 4-byte n_entries field
        self._u32be(0)                # placeholder — overwritten in _tg_close
        self._group_stack.append([pos, 0])

    def _tg_close(self):
        """End the most-recently opened TagGroup; back-patch the n_entries field."""
        pos, count = self._group_stack.pop()
        cur = self._b.tell()
        self._b.seek(pos)
        self._u32be(count)
        self._b.seek(cur)

    def _te_hdr(self, tag_type, name):
        nb = name.encode('ascii')
        self._byte(tag_type)
        self._u16be(len(nb))
        self._b.write(nb)
        if self._group_stack:            # tell parent group one more child was written
            self._group_stack[-1][1] += 1

    def _td_hdr(self, info):
        self._b.write(b'%%%%')
        self._u32be(len(info))
        for v in info: self._u32be(v)

    # ── scalar tag writers ────────────────────────────────────────────────────

    def _w_i16(self, name, v):
        self._te_hdr(21, name); self._td_hdr([2]); self._i16le(v)

    def _w_i32(self, name, v):
        self._te_hdr(21, name); self._td_hdr([3]); self._i32le(v)

    def _w_u32(self, name, v):
        self._te_hdr(21, name); self._td_hdr([5]); self._u32le(v)

    def _w_f32(self, name, v):
        self._te_hdr(21, name); self._td_hdr([6]); self._f32le(v)

    def _w_bool(self, name, v):
        self._te_hdr(21, name); self._td_hdr([8]); self._byte(1 if v else 0)

    def _w_u16(self, name, v):
        self._te_hdr(21, name); self._td_hdr([4]); self._u16le(v)

    # ── compound / array tag writers ─────────────────────────────────────────

    def _w_str(self, name, v):
        """String as ARRAY of uint16 — the ONLY valid DM3 string encoding."""
        chars = [ord(c) for c in str(v)]
        self._te_hdr(21, name)
        self._td_hdr([20, 4, len(chars)])
        for c in chars: self._u16le(c)

    def _w_color_i16(self, name, r, g, b):
        self._te_hdr(21, name)
        self._td_hdr([15, 0, 3, 0, 2, 0, 2, 0, 2])
        self._i16le(r); self._i16le(g); self._i16le(b)

    def _w_rect_i32(self, name, top, left, bottom, right):
        self._te_hdr(21, name)
        self._td_hdr([15, 0, 4, 0, 3, 0, 3, 0, 3, 0, 3])
        self._i32le(top); self._i32le(left)
        self._i32le(bottom); self._i32le(right)

    def _w_rect_f32(self, name, top, left, bottom, right):
        self._te_hdr(21, name)
        self._td_hdr([15, 0, 4, 0, 6, 0, 6, 0, 6, 0, 6])
        self._f32le(top); self._f32le(left)
        self._f32le(bottom); self._f32le(right)

    def _w_pt_f32(self, name, x, y):
        """2-coordinate float32 point (used for Offset, Scale transforms)."""
        self._te_hdr(21, name)
        self._td_hdr([15, 0, 2, 0, 6, 0, 6])
        self._f32le(x); self._f32le(y)

    def _w_pt_i32(self, name, x, y):
        """2-coordinate int32 point (used for SourceSize_Pixels)."""
        self._te_hdr(21, name)
        self._td_hdr([15, 0, 2, 0, 3, 0, 3])
        self._i32le(x); self._i32le(y)

    def _w_clut(self, name):
        """Greyscale CLUT: struct-array of 256 × {R,G,B} int16."""
        self._te_hdr(21, name)
        self._td_hdr([20, 15, 0, 3, 0, 2, 0, 2, 0, 2, 256])
        for i in range(256):
            self._i16le(i); self._i16le(i); self._i16le(i)

    def _w_bytes_array(self, name, data):
        """Array of raw uint8 bytes (info=[20,10,N])."""
        self._te_hdr(21, name)
        self._td_hdr([20, 10, len(data)])
        self._b.write(bytes(data))

    def _w_data(self, name, arr):
        """Float32 data array (info=[20,6,N])."""
        flat = np.ascontiguousarray(arr, dtype=np.float32).ravel()
        self._te_hdr(21, name)
        self._td_hdr([20, 6, len(flat)])
        self._b.write(flat.tobytes())

    def _w_data_i32(self, name, arr):
        """Int32 data array (info=[20,3,N]) — for RGBA survey image."""
        flat = np.ascontiguousarray(arr, dtype=np.int32).ravel()
        self._te_hdr(21, name)
        self._td_hdr([20, 3, len(flat)])
        self._b.write(flat.tobytes())

    # ── calibration helper ────────────────────────────────────────────────────

    def _cal_dim(self, key, origin, scale, units):
        self._te_hdr(20, key)
        self._tg_open(sorted_=True)
        self._w_f32('Origin', origin)
        self._w_f32('Scale',  scale)
        self._w_str('Units',  units)
        self._tg_close()

    # ── ImageData helper (shared by survey and SI) ────────────────────────────

    def _image_data_2d(self, survey_rgba, nx, ny):
        """Write ImageData group for the 2-D RGBA survey image."""
        self._te_hdr(20, 'ImageData')
        self._tg_open(sorted_=True)                # ── ImageData (5 children)

        # Calibrations ──────────────────────────────────────────────────────
        self._te_hdr(20, 'Calibrations')
        self._tg_open(sorted_=True)                # ── Calibrations (3 children)

        self._te_hdr(20, 'Brightness')
        self._tg_open(sorted_=True)                # ── Brightness (3 children)
        self._w_f32('Origin', 0.0)
        self._w_f32('Scale',  1.0)
        self._w_str('Units',  '')
        self._tg_close()                           # ── /Brightness

        self._te_hdr(20, 'Dimension')
        self._tg_open(sorted_=False)               # ── Dimension (2 children)
        self._cal_dim('0', 0.0, 1.0, '')           # X
        self._cal_dim('1', 0.0, 1.0, '')           # Y
        self._tg_close()                           # ── /Dimension

        self._w_bool('DisplayCalibratedUnits', True)
        self._tg_close()                           # ── /Calibrations

        # DataType 23 = RGBA color  ← must precede Data so readers know the type
        self._w_u32('DataType', 23)

        # Dimensions[0]=nx, [1]=ny  ← must precede Data so readers know the shape
        self._te_hdr(20, 'Dimensions')
        self._tg_open(sorted_=False)               # ── Dimensions (2 children)
        self._w_u32('', nx)
        self._w_u32('', ny)
        self._tg_close()                           # ── /Dimensions

        self._w_u32('PixelDepth', 4)

        # Data: RGBA int32 array, (ny, nx) C-order  ← written last
        self._w_data_i32('Data', survey_rgba)

        self._tg_close()                           # ── /ImageData

    def _image_data_3d(self, dm_data, nx, ny, ne,
                       cal_x, cal_y, cal_e, d_min, d_max):
        """Write ImageData group for the 3-D EELS SI."""
        self._te_hdr(20, 'ImageData')
        self._tg_open(sorted_=True)                # ── ImageData (5 children)

        # Calibrations ──────────────────────────────────────────────────────
        self._te_hdr(20, 'Calibrations')
        self._tg_open(sorted_=True)                # ── Calibrations (3 children)

        self._te_hdr(20, 'Brightness')
        self._tg_open(sorted_=True)                # ── Brightness (3 children)
        self._w_f32('Origin', 0.0)
        self._w_f32('Scale',  1.0)
        self._w_str('Units',  'Counts')
        self._tg_close()                           # ── /Brightness

        # Dimension: 3 axes, 0-indexed fastest→slowest: X, Y, E
        self._te_hdr(20, 'Dimension')
        self._tg_open(sorted_=False)               # ── Dimension (3 children)
        self._cal_dim('0', cal_x[0], cal_x[1], cal_x[2])  # X (fastest)
        self._cal_dim('1', cal_y[0], cal_y[1], cal_y[2])  # Y
        self._cal_dim('2', cal_e[0], cal_e[1], cal_e[2])  # E (slowest)
        self._tg_close()                           # ── /Dimension

        self._w_bool('DisplayCalibratedUnits', True)
        self._tg_close()                           # ── /Calibrations

        # DataType 2 = float32  ← must precede Data so readers know the type
        self._w_u32('DataType', 2)

        # Dimensions[0]=nx, [1]=ny, [2]=ne  ← must precede Data so readers know the shape
        self._te_hdr(20, 'Dimensions')
        self._tg_open(sorted_=False)               # ── Dimensions (3 children)
        self._w_u32('', nx)
        self._w_u32('', ny)
        self._w_u32('', ne)
        self._tg_close()                           # ── /Dimensions

        self._w_u32('PixelDepth', 4)

        # Data: (ne, ny, nx) C-order float32  ← written last
        self._w_data('Data', dm_data)

        self._tg_close()                           # ── /ImageData

    # ── main build method ─────────────────────────────────────────────────────

    def build(self, data_3d, cal_x, cal_y, cal_e, props=None, title='Spectrum Image', meta=None):
        """
        Build and return a complete GMS-compatible DM3 file as bytes.

        Parameters
        ----------
        data_3d : ndarray, shape (ny, nx, ne), float32
        cal_x   : (origin, scale, units_str)  — X spatial calibration
        cal_y   : (origin, scale, units_str)  — Y spatial calibration
        cal_e   : (origin, scale, units_str)  — energy-axis calibration
        props   : dict  — nhdf properties JSON (Nion metadata)
        title   : str   — image title string
        """
        self._b = BytesIO()
        props   = props or {}
        meta    = meta  or {}

        # ── meta access helpers with safe type coercion ───────────────────────
        def _mf(key, default=0.0):
            try:    return float(meta.get(key, default))
            except: return float(default)
        def _mi(key, default=0):
            try:    return int(meta.get(key, default))
            except: return int(default)
        def _ms(key, default=''):
            return str(meta.get(key, default))

        data_3d = np.asarray(data_3d, dtype=np.float32)
        ny, nx, ne = data_3d.shape

        # ── SI data: (ne, ny, nx) C-order  ────────────────────────────────────
        dm_data = np.ascontiguousarray(data_3d.transpose(2, 0, 1))

        d_min = float(data_3d.min())
        d_max = float(data_3d.max())
        if d_min == d_max:
            d_max = d_min + 1.0

        # ── Survey image: sum over energy, packed as RGBA int32 (DataType=23)
        survey_f = data_3d.sum(axis=2).astype(np.float32)   # (ny, nx)
        s_min, s_max = float(survey_f.min()), float(survey_f.max())
        if s_max > s_min:
            survey_u8 = ((survey_f - s_min) / (s_max - s_min) * 255
                         ).astype(np.uint32)
        else:
            survey_u8 = np.zeros((ny, nx), dtype=np.uint32)
        v = survey_u8
        # RGBA packed as int32 little-endian: byte0=R, byte1=G, byte2=B, byte3=A
        survey_rgba = (v | (v << 8) | (v << 16) | (np.uint32(255) << 24)
                       ).astype(np.int32)   # (ny, nx)

        # ── Random UniqueIDs ───────────────────────────────────────────────────
        uid0 = [random.getrandbits(32) for _ in range(4)]   # survey image
        uid1 = [random.getrandbits(32) for _ in range(4)]   # SI

        # ── Flatten nhdf metadata for Nion tag sub-group ───────────────────────
        def _flatten(d, prefix=''):
            out = []
            for k, v in d.items():
                full = f'{prefix}_{k}' if prefix else k
                if isinstance(v, dict):    out.extend(_flatten(v, full))
                elif isinstance(v, list):  out.append((full, json.dumps(v)))
                else:                      out.append((full, str(v)))
            return out

        nion_tags = [
            (k.replace('.', '_').replace(' ', '_'), str(v))
            for k, v in _flatten(props)
        ]
        n_nion = len(nion_tags)

        # ── build tag body into a temp buffer, then assemble ──────────────────
        main_buf = self._b
        self._b  = BytesIO()
        self._group_stack = []   # reset auto-count stack for each build()

        # display dimensions (float) used in several Rectangle/Bounds tags
        ny_f = float(ny)
        nx_f = float(nx)

        # ═════════════════════════════════════════════════════════════════════
        # ROOT TagGroup  (sorted=True, alphabetical; n_entries auto-counted)
        # ═════════════════════════════════════════════════════════════════════
        self._tg_open(sorted_=True)                # ── ROOT

        # ── 1. ApplicationBounds  (4 × int32 struct: top,left,bottom,right) ──
        self._w_rect_i32('ApplicationBounds', 0, 0, 895, 1512)

        # ── 2. DocumentObjectList ─────────────────────────────────────────────
        self._te_hdr(20, 'DocumentObjectList')
        self._tg_open(sorted_=False)               # ── DocumentObjectList

        # DocumentObjectList/[0]
        self._te_hdr(20, '')
        self._tg_open(sorted_=True)                # ── [0]

        # AnnotationGroupList  (empty)
        self._te_hdr(20, 'AnnotationGroupList')
        self._tg_open(sorted_=False)               # ── AnnotationGroupList
        self._tg_close()                           # ── /AnnotationGroupList

        self._w_i32('AnnotationType', 20)
        self._w_color_i16('BackgroundColor', 0, 0, 0)
        self._w_i16('BackgroundMode', 1)
        self._w_i16('FillMode', 1)
        self._w_color_i16('ForegroundColor', -1, -1, -1)
        self._w_bool('HasBackground', True)

        # ImageDisplayInfo
        self._te_hdr(20, 'ImageDisplayInfo')
        self._tg_open(sorted_=True)                # ── ImageDisplayInfo
        self._w_color_i16('BrightColor', -1, -1, -1)
        self._w_f32('Brightness', 0.5)
        self._w_bool('CaptionOn', False)
        self._w_u16('CaptionSize', 12)
        self._w_clut('CLUT')
        self._w_str('CLUTName', 'Greyscale')
        self._w_u32('ComplexMode', 4)
        self._w_f32('ComplexRange', 1000.0)
        self._w_f32('Contrast', 0.5)
        self._w_u32('ContrastMode', 1)
        # DimensionLabels (1 entry: empty-name string)
        self._te_hdr(20, 'DimensionLabels')
        self._tg_open(sorted_=False)               # ── DimensionLabels
        self._w_str('', '')
        self._tg_close()                           # ── /DimensionLabels
        self._w_bool('DoAutoSurvey', True)
        self._w_f32('EstimatedMax', float(d_max))
        self._w_f32('EstimatedMaxTrimPercentage', 0.001)
        self._w_f32('EstimatedMin', float(d_min))
        self._w_f32('EstimatedMinTrimPercentage', 0.001)
        self._w_f32('Gamma', 0.5)
        self._w_f32('HighLimit', float(d_max))
        self._w_f32('HiLimitContrastDeltaTriggerPercentage', 0.0)
        self._w_bool('IsInverted', False)
        self._w_f32('LowLimit', float(d_min))
        self._w_f32('LowLimitContrastDeltaTriggerPercentage', 0.0)
        # MainSliceId (1 entry: empty-name uint32 0)
        self._te_hdr(20, 'MainSliceId')
        self._tg_open(sorted_=False)               # ── MainSliceId
        self._w_u32('', 0)
        self._tg_close()                           # ── /MainSliceId
        self._w_f32('MinimumContrast', 0.0)
        self._w_f32('RangeAdjust', 1.0)
        self._w_u32('SparseSurvey_GridSize', 16)
        self._w_u32('SparseSurvey_NumberPixels', 32)
        self._w_bool('SparseSurvey_UseNumberPixels', True)
        self._w_u32('SurveyTechique', 2)    # note: typo matches reference
        self._tg_close()                           # ── /ImageDisplayInfo

        self._w_i32('ImageDisplayType', 1)
        self._w_u32('ImageSource', 0)       # → ImageSourceList[0]
        self._w_bool('IsMoveable', True)
        self._w_bool('IsResizable', True)
        self._w_bool('IsSelectable', True)
        self._w_bool('IsTranslatable', True)
        self._w_bool('IsVisible', True)

        # ObjectTags
        self._te_hdr(20, 'ObjectTags')
        self._tg_open(sorted_=True)                # ── ObjectTags
        self._w_bool('__is_not_copy', True)
        self._w_bool('__was_selected', False)
        self._tg_close()                           # ── /ObjectTags

        # Rectangle [top, left, bottom, right] in display pixels
        self._w_rect_f32('Rectangle', 0.0, 0.0, ny_f, nx_f)

        # UniqueID — scalar uint32 (not a group) matching ViewDisplayID=8
        self._w_u32('UniqueID', 8)
        self._tg_close()                           # ── /[0]
        self._tg_close()                           # ── /DocumentObjectList

        # ── 3. DocumentTags  (empty) ──────────────────────────────────────────
        self._te_hdr(20, 'DocumentTags')
        self._tg_open(sorted_=True)                # ── DocumentTags
        self._tg_close()                           # ── /DocumentTags

        # ── 4. HasWindowPosition ─────────────────────────────────────────────
        self._w_bool('HasWindowPosition', True)

        # ── 5. Image Behavior ────────────────────────────────────────────────
        self._te_hdr(20, 'Image Behavior')
        self._tg_open(sorted_=True)                # ── Image Behavior
        self._w_bool('DoIntegralZoom', False)
        self._w_rect_f32('ImageDisplayBounds', 0.0, 0.0, ny_f, nx_f)
        self._w_bool('IsZoomedToWindow', True)
        self._te_hdr(20, 'UnscaledTransform')
        self._tg_open(sorted_=True)                # ── UnscaledTransform
        self._w_pt_f32('Offset', 0.0, 0.0)
        self._w_pt_f32('Scale',  1.0, 1.0)
        self._tg_close()                           # ── /UnscaledTransform
        self._w_u32('ViewDisplayID', 8)
        self._w_rect_f32('WindowRect', 0.0, 0.0, ny_f, nx_f + 64.0)
        self._te_hdr(20, 'ZoomAndMoveTransform')
        self._tg_open(sorted_=True)                # ── ZoomAndMoveTransform
        self._w_pt_f32('Offset', 0.0, 0.0)
        self._w_pt_f32('Scale',  1.0, 1.0)
        self._tg_close()                           # ── /ZoomAndMoveTransform
        self._tg_close()                           # ── /Image Behavior

        # ── 6. ImageList ─────────────────────────────────────────────────────
        self._te_hdr(20, 'ImageList')
        self._tg_open(sorted_=False)               # ── ImageList

        # ── ImageList[0]  Survey image ────────────────────────────────────────
        self._te_hdr(20, '')
        self._tg_open(sorted_=True)                # ── ImageList[0]

        self._image_data_2d(survey_rgba, nx, ny)

        # ImageTags
        self._te_hdr(20, 'ImageTags')
        self._tg_open(sorted_=True)                # ── ImageTags (survey)
        self._te_hdr(20, 'GMS Version')
        self._tg_open(sorted_=True)                # ── GMS Version (survey)
        self._w_str('Created', '3.50.2819.0')
        self._tg_close()                           # ── /GMS Version (survey)
        self._tg_close()                           # ── /ImageTags (survey)

        self._w_str('Name', title + ' survey')

        # UniqueID (4 entries, sorted=False)
        self._te_hdr(20, 'UniqueID')
        self._tg_open(sorted_=False)               # ── UniqueID (survey)
        for u in uid0:
            self._w_u32('', u)
        self._tg_close()                           # ── /UniqueID (survey)
        self._tg_close()                           # ── /ImageList[0]

        # ── ImageList[1]  EELS Spectrum Image ────────────────────────────────
        self._te_hdr(20, '')
        self._tg_open(sorted_=True)                # ── ImageList[1]

        self._image_data_3d(dm_data, nx, ny, ne,
                            cal_x, cal_y, cal_e, d_min, d_max)

        # ImageTags  (sorted=True, alphabetical ASCII order)
        # Order: Acquisition(A) < GMS Version(G) < Meta Data(Me) <
        #        Microscope Info(Mi) < Nion(N) < SI(S,I=73) < Session Info(S,e=101)
        self._te_hdr(20, 'ImageTags')
        self._tg_open(sorted_=True)                # ── ImageTags (SI)

        # ── A: Acquisition ────────────────────────────────────────────────────
        self._te_hdr(20, 'Acquisition')
        self._tg_open(sorted_=True)                # ── Acquisition

        self._te_hdr(20, 'EELS')
        self._tg_open(sorted_=True)                # ── EELS

        # Acquisition/EELS/Acquisition  (C < E < I < N)
        self._te_hdr(20, 'Acquisition')
        self._tg_open(sorted_=True)                # ── EELS/Acquisition
        self._w_i32('Continuous mode',      4)
        self._w_f32('Exposure (s)',         _mf('exposure_s'))
        self._w_f32('Integration time (s)', _mf('integration_time_s'))
        self._w_i32('Number of frames',     _mi('frames', 1))
        self._tg_close()                           # ── /EELS/Acquisition

        # Acquisition/EELS/Experimental Conditions  (Col < Con < E)
        self._te_hdr(20, 'Experimental Conditions')
        self._tg_open(sorted_=True)                # ── Experimental Conditions
        self._w_f32('Collection semi-angle (mrad)',  _mf('collection_angle_mrad'))
        self._w_f32('Convergence semi-angle (mrad)', _mf('convergence_angle_mrad'))
        self._w_f32('Entrance aperture (mm)',         _mf('entrance_aperture_mm'))
        self._tg_close()                           # ── /Experimental Conditions

        self._tg_close()                           # ── /EELS
        self._tg_close()                           # ── /Acquisition

        # ── G: GMS Version ────────────────────────────────────────────────────
        self._te_hdr(20, 'GMS Version')
        self._tg_open(sorted_=True)                # ── GMS Version (SI)
        self._w_str('Created', '3.50.2819.0')
        self._w_str('Saved',   '3.50.2819.0')
        self._tg_close()                           # ── /GMS Version (SI)

        # ── Me: Meta Data — tells GMS this is an EELS Spectrum Image ──────────
        self._te_hdr(20, 'Meta Data')
        self._tg_open(sorted_=True)                # ── Meta Data
        self._w_str('Acquisition Mode', 'Parallel dispersive')
        self._w_str('Format',           'Spectrum image')   # lowercase 'i' — must match GMS
        self._w_str('Signal',           'EELS')
        self._tg_close()                           # ── /Meta Data

        # ── Mi: Microscope Info  (Ca < Cs < E < Ma < Mo < Probe C < Probe S < V)
        self._te_hdr(20, 'Microscope Info')
        self._tg_open(sorted_=True)                # ── Microscope Info
        self._w_f32('Camera Length (mm)',    _mf('camera_length_mm'))
        self._w_f32('Cs (mm)',               _mf('cs_mm'))
        self._w_f32('Emission Current (uA)', _mf('emission_current_ua'))
        self._w_f32('Magnification',         _mf('magnification'))
        self._w_str('Mode',                  _ms('mode'))
        self._w_f32('Probe Current (nA)',    _mf('probe_current_na'))
        self._w_f32('Probe Size (nm)',       _mf('probe_size_nm'))
        self._w_f32('Voltage',               _mf('beam_energy_kv', 200.0))
        self._tg_close()                           # ── /Microscope Info

        # ── N: Nion — nhdf properties flattened as string tags ────────────────
        self._te_hdr(20, 'Nion')
        self._tg_open(sorted_=False)               # ── Nion
        for tag_name, tag_val in nion_tags:
            self._w_str(tag_name, tag_val)
        self._tg_close()                           # ── /Nion

        # ── SI: STEM SI info  ('SI'=[83,73] < 'Se'=[83,101]) ─────────────────
        self._te_hdr(20, 'SI')
        self._tg_open(sorted_=True)                # ── SI
        self._te_hdr(20, 'Acquisition')
        self._tg_open(sorted_=True)                # ── SI/Acquisition
        self._w_f32('Pixel dwell time (s)', _mf('pixel_time_s'))
        self._w_str('Scan device',          _ms('scan_device'))
        self._tg_close()                           # ── /SI/Acquisition
        self._tg_close()                           # ── /SI

        # ── Se: Session Info  (C < M < O < S) ────────────────────────────────
        self._te_hdr(20, 'Session Info')
        self._tg_open(sorted_=True)                # ── Session Info
        self._w_str('Custom',     _ms('custom'))
        self._w_str('Microscope', _ms('microscope'))
        self._w_str('Operator',   _ms('operator'))
        self._w_str('Specimen',   _ms('specimen'))
        self._tg_close()                           # ── /Session Info

        self._tg_close()                           # ── /ImageTags (SI)

        self._w_str('Name', title)

        # UniqueID (4 entries, sorted=False)
        self._te_hdr(20, 'UniqueID')
        self._tg_open(sorted_=False)               # ── UniqueID (SI)
        for u in uid1:
            self._w_u32('', u)
        self._tg_close()                           # ── /UniqueID (SI)
        self._tg_close()                           # ── /ImageList[1]
        self._tg_close()                           # ── /ImageList

        # ── 7. ImageSourceList ───────────────────────────────────────────────
        self._te_hdr(20, 'ImageSourceList')
        self._tg_open(sorted_=False)               # ── ImageSourceList

        # ImageSourceList/[0]
        self._te_hdr(20, '')
        self._tg_open(sorted_=True)                # ── ImageSourceList[0]
        self._w_str('ClassName', 'ImageSource:Summed')   # ← key: NOT 'ImageSlice1D'
        self._w_bool('Do Sum', True)
        # Id (1 entry, sorted=False)
        self._te_hdr(20, 'Id')
        self._tg_open(sorted_=False)               # ── Id
        self._w_u32('', 0)
        self._tg_close()                           # ── /Id
        self._w_u32('ImageRef', 1)        # ← points to ImageList[1] (the SI)
        self._w_u32('LayerEnd',   0)
        self._w_u32('LayerStart', 0)
        self._w_u32('Summed Dimension', 2)  # ← collapse energy axis for map view
        self._tg_close()                           # ── /ImageSourceList[0]
        self._tg_close()                           # ── /ImageSourceList

        # ── 8. InImageMode ───────────────────────────────────────────────────
        self._w_bool('InImageMode', True)

        # ── 9. MinVersionList ────────────────────────────────────────────────
        self._te_hdr(20, 'MinVersionList')
        self._tg_open(sorted_=False)               # ── MinVersionList
        self._te_hdr(20, '')
        self._tg_open(sorted_=True)                # ── MinVersionList[0]
        self._w_u32('RequiredVersion', 50659328)
        self._tg_close()                           # ── /MinVersionList[0]
        self._tg_close()                           # ── /MinVersionList

        # ── 10. NextDocumentObjectID ─────────────────────────────────────────
        self._w_u32('NextDocumentObjectID', 10)

        # ── 11. Page Behavior ────────────────────────────────────────────────
        self._te_hdr(20, 'Page Behavior')
        self._tg_open(sorted_=True)                # ── Page Behavior
        self._w_bool('DoIntegralZoom', False)
        self._w_bool('DrawMargins', True)
        self._w_bool('DrawPaper', True)
        self._w_bool('IsFixedInPageMode', False)
        self._w_bool('IsZoomedToWindow', True)
        self._w_bool('LayedOut', False)
        self._te_hdr(20, 'PageTransform')
        self._tg_open(sorted_=True)                # ── PageTransform
        self._w_pt_f32('Offset', 0.0, 0.0)
        self._w_pt_f32('Scale',  1.0, 1.0)
        self._tg_close()                           # ── /PageTransform
        self._w_rect_f32('RestoreImageDisplayBounds', 0.0, 0.0, ny_f, nx_f)
        self._w_u32('RestoreImageDisplayID', 8)
        self._w_u32('TargetDisplayID', 0xFFFFFFFF)
        self._tg_close()                           # ── /Page Behavior

        # ── 12. PageSetup  (print layout settings) ───────────────────────────
        self._te_hdr(20, 'PageSetup')
        self._tg_open(sorted_=True)                # ── PageSetup
        # General: struct of 8 values (uint32 × 2, int32 × 6)
        self._te_hdr(21, 'General')
        self._td_hdr([15, 0, 8, 0, 5, 0, 5, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3])
        self._u32le(1);    self._u32le(1000)
        self._i32le(8500); self._i32le(11000)
        self._i32le(1000); self._i32le(1000)
        self._i32le(-1000); self._i32le(-1000)
        # Win32 / Win32_DevModeW / Win32_DevNamesW — opaque Windows print data
        self._w_bytes_array('Win32',           _WIN32_BYTES)
        self._w_bytes_array('Win32_DevModeW',  _WIN32_DEVMODEW_BYTES)
        self._w_bytes_array('Win32_DevNamesW', _WIN32_DEVNAMESW_BYTES)
        self._tg_close()                           # ── /PageSetup

        # ── 13. SentinelList  (empty) ────────────────────────────────────────
        self._te_hdr(20, 'SentinelList')
        self._tg_open(sorted_=False)               # ── SentinelList
        self._tg_close()                           # ── /SentinelList

        # ── 14. Thumbnails ───────────────────────────────────────────────────
        self._te_hdr(20, 'Thumbnails')
        self._tg_open(sorted_=False)               # ── Thumbnails
        self._te_hdr(20, '')
        self._tg_open(sorted_=True)                # ── Thumbnails[0]
        self._w_u32('ImageIndex', 0)
        self._w_pt_i32('SourceSize_Pixels', nx, ny)
        self._tg_close()                           # ── /Thumbnails[0]
        self._tg_close()                           # ── /Thumbnails

        # ── 15. WindowPosition  (4 × int32 struct: top,left,bottom,right) ────
        self._w_rect_i32('WindowPosition', 174, 399, 302, 591)

        self._tg_close()                           # ── /ROOT

        # ── assemble final file ───────────────────────────────────────────────
        tags_bytes = self._b.getvalue()
        self._b    = main_buf
        total      = 12 + len(tags_bytes)

        self._u32be(3)            # DM3 version
        self._u32be(total - 20)   # file_size field = total bytes - 20
        self._u32be(1)            # byte order (1 = little-endian data)
        self._b.write(tags_bytes)

        return self._b.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
#  NHDF READER
# ─────────────────────────────────────────────────────────────────────────────

def read_nhdf(path):
    """
    Read a Nion Swift NHDF spectrum-image file.

    Returns
    -------
    data   : ndarray  shape (ny, nx, ne), float32
    props  : dict     JSON metadata from the HDF5 'properties' attribute
    cal_x  : (origin, scale, units_str)  X spatial
    cal_y  : (origin, scale, units_str)  Y spatial
    cal_e  : (origin, scale, units_str)  energy axis
    title  : str
    """
    data  = None
    props = {}

    with h5py.File(path, 'r') as f:
        def _visit(name, obj):
            nonlocal data, props
            if isinstance(obj, h5py.Dataset) and obj.ndim == 3 and data is None:
                data = obj[()].astype(np.float32)
                raw  = obj.attrs.get('properties', None)
                if raw is not None:
                    s = raw if isinstance(raw, str) else raw.decode('utf-8', errors='replace')
                    try:
                        props = json.loads(s)
                    except Exception:
                        props = {}
        f.visititems(_visit)

    if data is None:
        raise ValueError('No 3-D dataset found in NHDF file.')

    dims = props.get('dimensional_calibrations', [])

    def _cal(i, default_units=''):
        d = dims[i] if i < len(dims) else {}
        return (float(d.get('offset', 0.0)),
                float(d.get('scale',  1.0)),
                str(d.get('units', default_units)))

    # NHDF shape (ny, nx, ne)
    cal_y = _cal(0, 'nm')
    cal_x = _cal(1, 'nm')
    cal_e = _cal(2, 'eV')

    title = props.get('title') or os.path.splitext(os.path.basename(path))[0]
    return data, props, cal_x, cal_y, cal_e, title


# ─────────────────────────────────────────────────────────────────────────────
#  DAT WRITER  (pure float32 bytes, no header)
# ─────────────────────────────────────────────────────────────────────────────

def write_raw(out_path, data_3d, cal_x=None, cal_y=None, cal_e=None, cal_i=None, meta=None):
    """
    Write raw float32 binary file in (ne, ny, nx) axis order — identical
    byte layout to the DM3 data array.

    Layout
    ------
    · dtype      : float32  (little-endian, 4 bytes/value)
    · axis order : C-order (ne slowest, nx fastest)  shape (ne, ny, nx)
    · no header, no footer, no padding
    · file size  = ne × ny × nx × 4 bytes exactly

    cal_x / cal_y / cal_e / cal_i : optional (offset, scale, units) tuples.
    When supplied they are recorded in the .txt sidecar.

    A .txt sidecar with the same base name is written to the same folder.
    """
    arr = np.asarray(data_3d, dtype=np.float32)
    ny, nx, ne = arr.shape
    data_f32 = np.ascontiguousarray(arr.transpose(2, 0, 1))  # → (ne, ny, nx)
    data_f32.tofile(out_path)

    # ── TXT sidecar ───────────────────────────────────────────────────────────
    base = os.path.splitext(out_path)[0]
    txt_path = base + '.txt'
    with open(txt_path, 'w') as fh:
        fh.write(f'DAT file: {os.path.basename(out_path)}\n')
        fh.write(f'Written : {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        fh.write('\n')
        fh.write('Format\n')
        fh.write('------\n')
        fh.write(f'dtype      : float32 (IEEE 754 single precision, little-endian)\n')
        fh.write(f'axis order : C-order  (ne slowest, nx fastest)\n')
        fh.write(f'shape      : ({ne}, {ny}, {nx})  =  ne x ny x nx\n')
        fh.write(f'values     : {ne * ny * nx:,}\n')
        fh.write(f'file size  : {ne * ny * nx * 4:,} bytes  '
                 f'({ne * ny * nx * 4 / 1024 / 1024:.3f} MB)\n')
        fh.write('\n')
        fh.write('Calibrations\n')
        fh.write('------------\n')
        fh.write(f'{"Axis":<14}{"dim":<6}{"offset (px)":<26}{"scale":<26}{"units"}\n')
        fh.write(f'{"-"*14}{"-"*6}{"-"*26}{"-"*26}{"-"*10}\n')
        _cals = [
            ('Y (rows)',  '0', cal_y),
            ('X (cols)',  '1', cal_x),
            ('Energy',    '2', cal_e),
            ('Intensity', '-', cal_i),
        ]
        for label, dim, cal in _cals:
            if cal is not None:
                _off_px = round(cal[0] / cal[1]) if cal[1] != 0.0 else 0
                fh.write(f'{label:<14}{dim:<6}{_off_px:<26}{cal[1]:<26.10g}{cal[2]}\n')
            else:
                fh.write(f'{label:<14}{dim:<6}{"n/a":<26}{"n/a":<26}n/a\n')
        # Energy axis: also show channel range explicitly in pixels
        if cal_e is not None and cal_e[1] != 0.0:
            _zlp_ch = round(-cal_e[0] / cal_e[1])
            fh.write(f'  Start channel : 0\n')
            fh.write(f'  End channel   : {ne - 1}\n')
            fh.write(f'  ZLP channel   : {_zlp_ch}\n')
        if meta:
            fh.write('\n')
            fh.write('Metadata\n')
            fh.write('--------\n')
            _meta_sections = [
                ('Microscope Info', [
                    ('Beam Energy (kV)',       'beam_energy_kv'),
                    ('Magnification',          'magnification'),
                    ('Camera Length (mm)',     'camera_length_mm'),
                    ('Cs (mm)',               'cs_mm'),
                    ('Emission Current (uA)', 'emission_current_ua'),
                    ('Probe Current (nA)',    'probe_current_na'),
                    ('Probe Size (nm)',       'probe_size_nm'),
                    ('Mode',                  'mode'),
                ]),
                ('Session Info', [
                    ('Specimen',   'specimen'),
                    ('Operator',   'operator'),
                    ('Microscope', 'microscope'),
                    ('Custom',     'custom'),
                ]),
                ('EELS', [
                    ('Convergence semi-angle (mrad)', 'convergence_angle_mrad'),
                    ('Collection semi-angle (mrad)',  'collection_angle_mrad'),
                    ('Entrance Aperture (mm)',        'entrance_aperture_mm'),
                    ('Exposure (s)',                  'exposure_s'),
                    ('Frames Summed',                 'frames'),
                    ('Integration Time (s)',          'integration_time_s'),
                ]),
                ('STEM SI', [
                    ('Pixel Time (s)', 'pixel_time_s'),
                    ('Scan Device',   'scan_device'),
                ]),
            ]
            for section, fields in _meta_sections:
                fh.write(f'  {section}\n')
                for label, key in fields:
                    fh.write(f'    {label:<34}: {meta.get(key, "")}\n')
                fh.write('\n')

        fh.write('Read back with Python / NumPy\n')
        fh.write('-----------------------------\n')
        fh.write('import numpy as np\n')
        fh.write(f'd = np.fromfile("{os.path.basename(out_path)}", dtype="<f4")'
                 f'.reshape({ne}, {ny}, {nx})\n')
        fh.write('# d[energy_index, y_index, x_index]\n')
        fh.write(f'# d.shape == ({ne}, {ny}, {nx})\n')

    return data_f32.nbytes, txt_path


# ─────────────────────────────────────────────────────────────────────────────
#  DM3 PATCHER  (preserves every byte of an existing DM3 except the SI data)
# ─────────────────────────────────────────────────────────────────────────────

def _dm3_find_si_data(blob):
    """
    Walk a DM3 binary and return (offset, nbytes) of the largest
    float32 'Data' array tag — the SI data at ImageList[1].ImageData.Data.
    Raises ValueError if not found.
    """
    import struct as _st
    p = [12]   # mutable position pointer; skip 12-byte file header

    ESIZES = {2:2, 3:4, 4:2, 5:4, 6:4, 7:8, 8:1, 9:1, 10:1, 11:8, 12:8}

    def rb(n):    v = blob[p[0]:p[0]+n]; p[0] += n; return v
    def ru8():    v = blob[p[0]];        p[0] += 1; return v
    def ri16be(): v,=_st.unpack_from('>h', blob, p[0]); p[0]+=2; return v
    def ri32be(): v,=_st.unpack_from('>i', blob, p[0]); p[0]+=4; return v
    def ru32be(): v,=_st.unpack_from('>I', blob, p[0]); p[0]+=4; return v

    best = [None]   # will hold (offset, nbytes) of the winning candidate

    def skip_val(descs):
        enc = descs[0]
        if enc == 20:                   # array
            etype = descs[1]
            alen  = descs[-1]
            if etype == 15:             # struct array
                nf  = descs[3]
                esz = sum(ESIZES.get(descs[4 + i*2 + 1], 0) for i in range(nf))
            else:
                esz = ESIZES.get(etype, 0)
            p[0] += esz * alen
        elif enc == 15:                 # struct
            nf    = descs[2]
            total = sum(ESIZES.get(descs[3 + i*2 + 1], 0) for i in range(nf))
            p[0] += total
        elif enc == 18:                 # string  (uint32-BE len, then len × uint16)
            n = ru32be(); p[0] += n * 2
        else:
            p[0] += ESIZES.get(enc, 0)

    def walk_group():
        ru8(); ru8()                    # sorted / open flag bytes
        n = ru32be()                    # entry count (BE uint32)
        for _ in range(n):
            walk_entry()

    def walk_entry():
        ttype = ru8()
        nl    = ri16be()
        name  = rb(nl).decode('latin-1', errors='replace')
        if ttype == 20:
            walk_group()
        elif ttype == 21:
            magic = rb(4)
            if magic != b'%%%%':
                raise ValueError(f'Bad magic {magic!r} at offset {p[0]-4}')
            nd    = ri32be()
            descs = [ri32be() for _ in range(nd)]
            if not descs:
                return
            # Capture the largest float32 array tag named 'Data'
            if (name == 'Data' and len(descs) == 3
                    and descs[0] == 20 and descs[1] == 6):
                offset = p[0]
                nbytes = descs[2] * 4
                if best[0] is None or nbytes > best[0][1]:
                    best[0] = (offset, nbytes)
                p[0] += nbytes
            else:
                skip_val(descs)

    walk_group()
    if best[0] is None:
        raise ValueError('No float32 Data array found in DM3 file.')
    return best[0]   # (offset, nbytes)


def patch_dm3_si_data(ref_dm3_path, data_3d):
    """
    Build a new DM3 by copying every byte of *ref_dm3_path* and replacing
    only the SI float32 data array with the contents of *data_3d*.

    All calibrations, metadata, UniqueIDs, and structural tags are taken
    verbatim from the reference file — nothing is regenerated.

    Parameters
    ----------
    ref_dm3_path : str      — path to the existing DM3 that supplies metadata
    data_3d      : ndarray, shape (ny, nx, ne), float32 — new SI data

    Returns
    -------
    bytes  — patched DM3 file ready to write to disk
    """
    with open(ref_dm3_path, 'rb') as fh:
        blob = bytearray(fh.read())

    offset, nbytes = _dm3_find_si_data(bytes(blob))

    arr     = np.asarray(data_3d, dtype=np.float32)
    ny, nx, ne = arr.shape
    dm_data = np.ascontiguousarray(arr.transpose(2, 0, 1))   # → (ne, ny, nx)

    if dm_data.nbytes != nbytes:
        raise ValueError(
            f'Data size mismatch: nhdf yields {dm_data.nbytes} bytes '
            f'but the reference DM3 has {nbytes} bytes at the data slot  '
            f'(nhdf shape {arr.shape}  ≠  DM3 expects {nbytes // 4} elements).')

    blob[offset:offset + nbytes] = dm_data.tobytes()
    return bytes(blob)


# ─────────────────────────────────────────────────────────────────────────────
#  DM3 FULL PATCHER  (data + calibrations + title from NHDF)
# ─────────────────────────────────────────────────────────────────────────────

def _dm3_collect_records(blob):
    """
    Walk a DM3 binary and return a list of every leaf tag:
        (path, type_str, value, byte_offset, byte_size, aux)
    where aux = element count (for str_u16 arrays) or None.
    Uses [N] bracket notation for anonymous (empty-named) group entries so
    that e.g. ImageList[1].ImageData.Calibrations.Dimension[2].Origin is
    unambiguous.
    """
    import struct as _st
    ESIZES = {2:2, 3:4, 4:2, 5:4, 6:4, 7:8, 8:1, 9:1, 10:1, 11:8, 12:8}
    p = [12]   # position pointer; skip 12-byte DM3 file header
    records = []

    def rb(n):    v = bytes(blob[p[0]:p[0]+n]); p[0] += n; return v
    def ru8():    v = blob[p[0]];               p[0] += 1; return v
    def ri16be(): v,=_st.unpack_from('>h',blob,p[0]); p[0]+=2; return v
    def ri32be(): v,=_st.unpack_from('>i',blob,p[0]); p[0]+=4; return v
    def ru32be(): v,=_st.unpack_from('>I',blob,p[0]); p[0]+=4; return v

    def record_val(descs, path):
        enc = descs[0]
        if enc == 20:                               # typed array
            etype = descs[1]; alen = descs[-1]
            if etype == 15:
                nf  = descs[3]
                esz = sum(ESIZES.get(descs[4+i*2+1], 0) for i in range(nf))
            else:
                esz = ESIZES.get(etype, 0)
            off = p[0]; sz = esz * alen; p[0] += sz
            if etype == 4:                          # uint16 string array
                try:  v = bytes(blob[off:off+sz]).decode('utf-16-le').rstrip('\x00')
                except: v = ''
                records.append((path, 'str_u16', v, off, sz, alen))
        elif enc == 15:                             # struct
            nf    = descs[2]
            total = sum(ESIZES.get(descs[3+i*2+1], 0) for i in range(nf))
            p[0] += total
        elif enc == 18:                             # variable-length string
            n = ru32be(); off = p[0]; sz = n*2; p[0] += sz
            try:  v = bytes(blob[off:off+sz]).decode('utf-16-le').rstrip('\x00')
            except: v = ''
            records.append((path, 'str18', v, off, sz, n))
        elif enc == 6:
            off=p[0]; v,=_st.unpack_from('<f',blob,p[0]); p[0]+=4
            records.append((path,'f32',v,off,4,None))
        elif enc == 7:
            off=p[0]; v,=_st.unpack_from('<d',blob,p[0]); p[0]+=8
            records.append((path,'f64',v,off,8,None))
        elif enc == 3:
            off=p[0]; v,=_st.unpack_from('<i',blob,p[0]); p[0]+=4
            records.append((path,'i32',v,off,4,None))
        elif enc == 4:
            off=p[0]; v,=_st.unpack_from('<H',blob,p[0]); p[0]+=2
            records.append((path,'u16',v,off,2,None))
        elif enc == 5:
            off=p[0]; v,=_st.unpack_from('<I',blob,p[0]); p[0]+=4
            records.append((path,'u32',v,off,4,None))
        elif enc == 8:
            off=p[0]; v=blob[p[0]]; p[0]+=1
            records.append((path,'bool8',v,off,1,None))
        elif enc == 2:
            off=p[0]; v,=_st.unpack_from('<h',blob,p[0]); p[0]+=2
            records.append((path,'i16',v,off,2,None))
        else:
            p[0] += ESIZES.get(enc, 0)

    def walk_group(path):
        ru8(); ru8()            # sorted / open flags
        n = ru32be()
        anon = [0]              # counter for unnamed children only
        for _ in range(n):
            walk_entry(path, anon)

    def walk_entry(parent, anon):
        ttype = ru8()
        nl    = ri16be()
        name  = rb(nl).decode('latin-1', 'replace')
        if name:
            full = f'{parent}.{name}' if parent else name
        else:
            full = f'{parent}[{anon[0]}]'
            anon[0] += 1
        if ttype == 20:
            walk_group(full)
        elif ttype == 21:
            magic = rb(4)
            if magic != b'%%%%':
                raise ValueError(f'DM3 bad magic {magic!r} at {p[0]-4}')
            nd    = ri32be()
            descs = [ri32be() for _ in range(nd)]
            if not descs:
                return
            # SI data: large float32 array named 'Data'
            if (name == 'Data' and len(descs) == 3
                    and descs[0] == 20 and descs[1] == 6):
                records.append((full, 'SI_DATA', None, p[0], descs[2]*4, descs[2]))
                p[0] += descs[2] * 4
            else:
                record_val(descs, full)

    walk_group('')
    return records


def patch_dm3_full(ref_dm3_path, data_3d, cal_x, cal_y, cal_e,
                   cal_i=None, title=''):
    """
    Build a patched DM3 by:
      1. Copying every byte of *ref_dm3_path*  (preserves GMS tag structure)
      2. Replacing the SI float32 data array with *data_3d*
      3. Overwriting calibrations and title with the supplied NHDF values

    DM3 Dimension axis mapping (C-order, fastest axis first):
      Dimension[0] = X axis  (fastest)   <- cal_x
      Dimension[1] = Y axis              <- cal_y
      Dimension[2] = Energy axis         <- cal_e

    DM3 energy-Origin convention (GMS):
      stored as  -cal_e[0] / cal_e[1]   (channel index of zero energy)
    Spatial Origin: stored as 0.0  (GMS relative-image convention)
    """
    import struct as _st, re as _re

    with open(ref_dm3_path, 'rb') as fh:
        blob = bytearray(fh.read())

    # ── 1. Replace SI data ────────────────────────────────────────────────────
    data_off, data_nb = _dm3_find_si_data(bytes(blob))
    arr     = np.asarray(data_3d, dtype=np.float32)
    ny, nx, ne = arr.shape
    dm_data = np.ascontiguousarray(arr.transpose(2, 0, 1))   # (ne, ny, nx)
    if dm_data.nbytes != data_nb:
        raise ValueError(
            f'Data size mismatch: nhdf yields {dm_data.nbytes} B '
            f'but reference DM3 slot has {data_nb} B '
            f'(nhdf shape {arr.shape}).')
    blob[data_off:data_off + data_nb] = dm_data.tobytes()

    # ── 2. Collect all tag locations ──────────────────────────────────────────
    records  = _dm3_collect_records(bytes(blob))
    by_path  = {r[0]: r for r in records}

    # ── helpers ───────────────────────────────────────────────────────────────
    def patch_f32(path, value):
        r = by_path.get(path)
        if r and r[1] == 'f32':
            _st.pack_into('<f', blob, r[3], float(value))

    def patch_str(path, new_str):
        """In-place overwrite of a str_u16 slot; skips 0-length slots."""
        r = by_path.get(path)
        if r and r[1] == 'str_u16' and r[4] > 0:
            off, sz_bytes, alen = r[3], r[4], r[5]
            chars = new_str[:alen].ljust(alen, '\x00')
            blob[off:off + sz_bytes] = chars.encode('utf-16-le')

    # ── 3. Identify the ImageList index that holds the SI data ────────────────
    si_il = 1   # default
    for r in records:
        if r[1] == 'SI_DATA':
            m = _re.match(r'ImageList\[(\d+)\]', r[0])
            if m:
                si_il = int(m.group(1))
            break
    IL = f'ImageList[{si_il}]'

    # ── 4. Patch axis calibrations ────────────────────────────────────────────
    # DM3 Dimension[0]=X (fastest), [1]=Y, [2]=Energy
    for di, cal in ((0, cal_x), (1, cal_y), (2, cal_e)):
        if cal is None:
            continue
        base = f'{IL}.ImageData.Calibrations.Dimension[{di}]'
        # Energy origin = zero-loss channel; spatial origin = 0 (GMS convention)
        origin = float(round(-cal[0] / cal[1])) if (di == 2 and cal[1] != 0) else 0.0
        patch_f32(f'{base}.Origin', origin)
        patch_f32(f'{base}.Scale',  cal[1])
        patch_str(f'{base}.Units',  cal[2])

    # ── 5. Patch brightness (intensity) calibration ───────────────────────────
    if cal_i is not None:
        base = f'{IL}.ImageData.Calibrations.Brightness'
        patch_f32(f'{base}.Origin', cal_i[0])
        patch_f32(f'{base}.Scale',  cal_i[1])
        patch_str(f'{base}.Units',  cal_i[2])

    # ── 6. Patch Processing history energy calibration (if present) ───────────
    if cal_e is not None:
        for r in records:
            if not r[0].startswith(IL):
                continue
            rp = r[0]
            if r[1] == 'f32':
                if rp.endswith('.new origin') or rp.endswith('.old origin'):
                    _st.pack_into('<f', blob, r[3], float(cal_e[0]))
                elif rp.endswith('.new scale') or rp.endswith('.old scale'):
                    _st.pack_into('<f', blob, r[3], float(cal_e[1]))
            elif r[1] == 'str_u16' and r[4] > 0:
                if rp.endswith('.new units') or rp.endswith('.old units'):
                    off, sz, alen = r[3], r[4], r[5]
                    chars = cal_e[2][:alen].ljust(alen, '\x00')
                    blob[off:off + sz] = chars.encode('utf-16-le')

    # ── 7. Update image title ─────────────────────────────────────────────────
    if title:
        patch_str(f'{IL}.Name', title)

    return bytes(blob)


def build_dm3_from_nhdf(data_3d, cal_x, cal_y, cal_e, cal_i=None, title=''):
    """
    Build a complete, GMS-compatible DM3 Spectrum Image from scratch.

    Uses the embedded _DM3_TEMPLATE_B64 / _DM3_TEMPLATE_HEAD_SIZE constants
    — no external reference file is needed.

    Parameters
    ----------
    data_3d : ndarray  shape (ny, nx, ne), float32
    cal_x   : (origin, scale, units)   X spatial calibration
    cal_y   : (origin, scale, units)   Y spatial calibration
    cal_e   : (origin, scale, units)   energy calibration
    cal_i   : (origin, scale, units)   intensity calibration  (optional)
    title   : str                      image title

    DM3 Dimension axis mapping (C-order, fastest first):
      Dimension[0] = X  <- cal_x
      Dimension[1] = Y  <- cal_y
      Dimension[2] = E  <- cal_e
    Energy Origin stored as -cal_e[0]/cal_e[1]  (zero-loss channel).
    Spatial Origin stored as 0.0  (GMS relative convention).
    """
    import struct as _st, zlib as _zl, base64 as _b64, re as _re

    arr     = np.asarray(data_3d, dtype=np.float32)
    ny, nx, ne = arr.shape
    dm_data = np.ascontiguousarray(arr.transpose(2, 0, 1))   # (ne, ny, nx)

    # ── 1. Decompress template; split head and tail ───────────────────────────
    tmpl = _zl.decompress(_b64.b64decode(_DM3_TEMPLATE_B64))
    hs   = _DM3_TEMPLATE_HEAD_SIZE
    head = bytearray(tmpl[:hs])
    tail = bytes(tmpl[hs:])

    # ── 2. Write data count into head (big-endian int32, last 4 bytes) ────────
    _st.pack_into('>i', head, hs - 4, ny * nx * ne)

    # ── 3. Assemble full DM3 ──────────────────────────────────────────────────
    total = len(head) + dm_data.nbytes + len(tail)
    blob  = bytearray(total)
    blob[:hs]                        = head
    blob[hs:hs + dm_data.nbytes]     = dm_data.tobytes()
    blob[hs + dm_data.nbytes:]       = tail

    # ── 4. Update file_size_field in DM3 12-byte header (offset 4, BE uint32) ─
    _st.pack_into('>I', blob, 4, total - 20)

    # ── 5. Collect all tag locations ──────────────────────────────────────────
    records = _dm3_collect_records(bytes(blob))
    by_path = {r[0]: r for r in records}

    # ── patch helpers ─────────────────────────────────────────────────────────
    def patch_f32(path, value):
        r = by_path.get(path)
        if r and r[1] == 'f32':
            _st.pack_into('<f', blob, r[3], float(value))

    def patch_u32(path, value):
        r = by_path.get(path)
        if r and r[1] == 'u32':
            _st.pack_into('<I', blob, r[3], int(value))

    def patch_str(path, new_str):
        r = by_path.get(path)
        if r and r[1] == 'str_u16' and r[4] > 0:
            off, sz_bytes, alen = r[3], r[4], r[5]
            chars = new_str[:alen].ljust(alen, '\x00')
            blob[off:off + sz_bytes] = chars.encode('utf-16-le')

    # ── 6. Find ImageList index that holds the SI data ────────────────────────
    si_il = 1
    for r in records:
        if r[1] == 'SI_DATA':
            m = _re.match(r'ImageList\[(\d+)\]', r[0])
            if m:
                si_il = int(m.group(1))
            break
    IL = f'ImageList[{si_il}]'

    # ── 7. Patch data shape (Dimensions: fastest-first = nx, ny, ne) ──────────
    patch_u32(f'{IL}.ImageData.Dimensions[0]', nx)
    patch_u32(f'{IL}.ImageData.Dimensions[1]', ny)
    patch_u32(f'{IL}.ImageData.Dimensions[2]', ne)

    # ── 8. Patch axis calibrations ────────────────────────────────────────────
    for di, cal in ((0, cal_x), (1, cal_y), (2, cal_e)):
        if cal is None:
            continue
        base = f'{IL}.ImageData.Calibrations.Dimension[{di}]'
        origin = float(round(-cal[0] / cal[1])) if (di == 2 and cal[1] != 0) else 0.0
        patch_f32(f'{base}.Origin', origin)
        patch_f32(f'{base}.Scale',  cal[1])
        patch_str(f'{base}.Units',  cal[2])

    # ── 9. Patch brightness (intensity) calibration ───────────────────────────
    if cal_i is not None:
        base = f'{IL}.ImageData.Calibrations.Brightness'
        patch_f32(f'{base}.Origin', cal_i[0])
        patch_f32(f'{base}.Scale',  cal_i[1])
        patch_str(f'{base}.Units',  cal_i[2])

    # ── 10. Patch Processing history energy calibration (if present) ──────────
    if cal_e is not None:
        for r in records:
            if not r[0].startswith(IL):
                continue
            rp = r[0]
            if r[1] == 'f32':
                if rp.endswith('.new origin') or rp.endswith('.old origin'):
                    _st.pack_into('<f', blob, r[3], float(cal_e[0]))
                elif rp.endswith('.new scale') or rp.endswith('.old scale'):
                    _st.pack_into('<f', blob, r[3], float(cal_e[1]))
            elif r[1] == 'str_u16' and r[4] > 0:
                if rp.endswith('.new units') or rp.endswith('.old units'):
                    off, sz, alen = r[3], r[4], r[5]
                    chars = cal_e[2][:alen].ljust(alen, '\x00')
                    blob[off:off + sz] = chars.encode('utf-16-le')

    # ── 11. Update image title ────────────────────────────────────────────────
    if title:
        patch_str(f'{IL}.Name', title)

    # ── 11b. Patch 'Start Energy' / 'End Energy' → pixel (channel) units ───────
    # Overwrite the scalar value in-place; no bytes are added or removed so the
    # file structure stays intact and GMS will not see any corruption.
    for _tag_name, _px_val in (('Start Energy', 0), ('End Energy', ne - 1)):
        _nb     = _tag_name.encode('ascii')
        _needle = bytes([21]) + _st.pack('>H', len(_nb)) + _nb + b'%%%%'
        _pos    = bytes(blob).find(_needle)
        if _pos < 0:
            continue
        _p   = _pos + 1 + 2 + len(_nb) + 4         # after type + namelen + name + magic
        _nd  = _st.unpack_from('>i', blob, _p)[0]   # descriptor count
        _enc = _st.unpack_from('>i', blob, _p + 4)[0]  # first descriptor = encoding
        _p  += 4 + _nd * 4                           # advance past ndesc field + descriptors
        if   _enc == 5:  _st.pack_into('<f', blob, _p, float(_px_val))   # float32
        elif _enc == 12: _st.pack_into('<d', blob, _p, float(_px_val))   # float64
        elif _enc == 3:  _st.pack_into('<i', blob, _p, int(_px_val))     # int32
        elif _enc == 8:  _st.pack_into('<I', blob, _p, int(_px_val))     # uint32

    # ── 12. Insert 'Acquisition Mode' into Meta Data (template may not have it)
    # Meta Data is sorted=True; alphabetical order: Acquisition Mode(A) < Format(F) < Signal(S)
    # Strategy: locate the 'Format' tag entry, insert 'Acquisition Mode' before it,
    # increment the Meta Data group entry count, re-patch file_size_field.
    #
    # The template may encode strings as type-18 (variable-length, 1 descriptor)
    # OR type-20/etype-4 (uint16 array, 3 descriptors).  The number of bytes
    # before the string data differs, so we handle both cases.
    _md_fmt = by_path.get(f'{IL}.ImageTags.Meta Data.Format')
    if _md_fmt is None:
        # Fall back: scan every record for Meta Data.Format (any ImageList index)
        for _r in records:
            if _r[0].endswith('.ImageTags.Meta Data.Format'):
                _md_fmt = _r
                break
    if _md_fmt and _md_fmt[1] in ('str_u16', 'str18'):
        _fmt_str_off  = _md_fmt[3]            # byte offset where string DATA begins
        _fmt_name_len = len(b'Format')        # 6
        if _md_fmt[1] == 'str_u16':
            # type(1)+namelen(2)+name+magic(4)+ndesc(4)+3*desc(12)
            _hdr_sz = 1 + 2 + _fmt_name_len + 4 + 4 + 12
        else:                                 # str18: 1 descriptor + charcount field
            # type(1)+namelen(2)+name+magic(4)+ndesc(4)+1*desc(4)+charcount(4)
            _hdr_sz = 1 + 2 + _fmt_name_len + 4 + 4 + 4 + 4
        _fmt_tag_start = _fmt_str_off - _hdr_sz
        # Meta Data group header = sorted(1)+open(1)+count(4BE); count is last 4 bytes
        _count_off     = _fmt_tag_start - 4

        _name_b = b'Acquisition Mode'          # 16 bytes
        _val    = 'Parallel dispersive'        # 19 chars
        _chars  = _val.encode('utf-16-le')
        _n      = len(_val)
        _tag    = bytearray()
        _tag.append(21)                        # TagData
        _tag   += _st.pack('>H', len(_name_b))# name length (uint16 BE)
        _tag   += _name_b                      # name bytes
        _tag   += b'%%%%'                      # DM3 magic
        _tag   += _st.pack('>i', 3)            # 3 type descriptors
        _tag   += _st.pack('>i', 20)           # array type
        _tag   += _st.pack('>i', 4)            # uint16 element type
        _tag   += _st.pack('>i', _n)           # element count (19)
        _tag   += _chars                       # 'Parallel dispersive' as uint16-LE

        blob[_fmt_tag_start:_fmt_tag_start] = _tag          # insert before Format
        _curr = _st.unpack_from('>I', blob, _count_off)[0]
        _st.pack_into('>I', blob, _count_off, _curr + 1)    # count: 2 → 3
        _st.pack_into('>I', blob, 4, len(blob) - 20)        # re-patch file_size_field

    return bytes(blob)


# ─────────────────────────────────────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('NHDF Spectrum Image Converter  v0.5')
        self.resizable(False, False)
        self._fmt  = tk.StringVar(value='dm3')
        self._mode = tk.StringVar(value='single')

        # ── Metadata vars (written into DM3 ImageTags groups) ─────────────────
        self._meta = {
            # Microscope Info tab
            'beam_energy_kv':          tk.DoubleVar(value=200.0),
            'magnification':           tk.DoubleVar(value=0.0),
            'camera_length_mm':        tk.DoubleVar(value=0.0),
            'cs_mm':                   tk.DoubleVar(value=0.0),
            'emission_current_ua':     tk.DoubleVar(value=0.0),
            'probe_current_na':        tk.DoubleVar(value=0.0),
            'probe_size_nm':           tk.DoubleVar(value=0.0),
            'mode':                    tk.StringVar(value=''),
            # Session Info tab
            'specimen':                tk.StringVar(value=''),
            'operator':                tk.StringVar(value=''),
            'microscope':              tk.StringVar(value=''),
            'custom':                  tk.StringVar(value=''),
            # EELS tab
            'convergence_angle_mrad':  tk.DoubleVar(value=0.0),
            'collection_angle_mrad':   tk.DoubleVar(value=0.0),
            'entrance_aperture_mm':    tk.DoubleVar(value=0.0),
            'exposure_s':              tk.DoubleVar(value=0.0),
            'frames':                  tk.IntVar(value=1),
            'integration_time_s':      tk.DoubleVar(value=0.0),
            # STEM SI tab
            'pixel_time_s':            tk.DoubleVar(value=0.0),
            'scan_device':             tk.StringVar(value=''),
        }
        self._build_ui()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        P = 12

        # Mode selector
        fm_mode = ttk.LabelFrame(self, text='Export Mode', padding=P)
        fm_mode.grid(row=0, column=0, sticky='ew', padx=P, pady=(P, 4))
        ttk.Radiobutton(fm_mode, text='Single file',   variable=self._mode,
                        value='single', command=self._on_mode_changed).grid(row=0, column=0, padx=(0, 30))
        ttk.Radiobutton(fm_mode, text='Batch folder',  variable=self._mode,
                        value='batch',  command=self._on_mode_changed).grid(row=0, column=1)

        # Input
        self._fi_frame = ttk.LabelFrame(self, text='Input File  (.nhdf)', padding=P)
        self._fi_frame.grid(row=1, column=0, sticky='ew', padx=P, pady=4)
        self._in_var = tk.StringVar()
        self._in_var.trace_add('write', self._on_in_changed)
        ttk.Entry(self._fi_frame, textvariable=self._in_var, width=58).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(self._fi_frame, text='Browse…', command=self._browse_in).grid(row=0, column=1)

        # Format selector
        ff = ttk.LabelFrame(self, text='Output Format  (click to select)', padding=P)
        ff.grid(row=2, column=0, sticky='ew', padx=P, pady=4)

        self._btn_dm3 = tk.Button(
            ff, width=30, height=4, font=('Segoe UI', 9),
            text='DM3\nGatan Digital Micrograph 3\n(Spectrum Image + calibrations)',
            command=lambda: self._sel('dm3'))
        self._btn_dm3.grid(row=0, column=0, padx=10, pady=4)

        self._btn_raw = tk.Button(
            ff, width=30, height=4, font=('Segoe UI', 9),
            text='DAT\nPure float32 binary\n(ne × ny × nx, no header)',
            command=lambda: self._sel('dat'))
        self._btn_raw.grid(row=0, column=1, padx=10, pady=4)

        self._sel('dm3')

        # Output
        fo = ttk.LabelFrame(self, text='Output', padding=P)
        fo.grid(row=3, column=0, sticky='ew', padx=P, pady=4)
        fo.columnconfigure(1, weight=1)
        self._out_dir_var    = tk.StringVar()
        self._out_name_var   = tk.StringVar()
        self._out_folder_var = tk.StringVar()

        # Single-mode widgets
        self._lbl_path    = ttk.Label(fo, text='Path:')
        self._ent_path    = ttk.Entry(fo, textvariable=self._out_dir_var,  width=55)
        self._btn_save_as = ttk.Button(fo, text='Save As…', command=self._browse_out)
        self._lbl_file    = ttk.Label(fo, text='File:')
        self._ent_file    = ttk.Entry(fo, textvariable=self._out_name_var, width=55)
        self._lbl_path.grid(   row=0, column=0, sticky='w',  padx=(0, 4))
        self._ent_path.grid(   row=0, column=1, sticky='ew', padx=(0, 6))
        self._btn_save_as.grid(row=0, column=2)
        self._lbl_file.grid(   row=1, column=0, sticky='w',  padx=(0, 4), pady=(4, 0))
        self._ent_file.grid(   row=1, column=1, sticky='ew', pady=(4, 0))

        # Batch-mode widgets (hidden until batch mode is selected)
        self._lbl_out_folder        = ttk.Label(fo, text='Output Folder:')
        self._ent_out_folder        = ttk.Entry(fo, textvariable=self._out_folder_var, width=55)
        self._btn_browse_out_folder = ttk.Button(fo, text='Browse…',
                                                 command=self._browse_out_folder)

        # Metadata panel (written to DM3 ImageTags groups)
        fmeta = ttk.LabelFrame(self, text='Metadata  (written to DM3 tag groups)', padding=P)
        fmeta.grid(row=4, column=0, sticky='ew', padx=P, pady=4)
        self._build_meta_panel(fmeta)

        # Convert button + progress
        self._btn_conv = ttk.Button(self, text='  Convert  ', command=self._convert)
        self._btn_conv.grid(row=5, column=0, pady=8)

        self._prog = ttk.Progressbar(self, mode='indeterminate', length=560)
        self._prog.grid(row=6, column=0, padx=P, pady=2)

        # Log
        fl = ttk.LabelFrame(self, text='Status / Log', padding=4)
        fl.grid(row=7, column=0, sticky='nsew', padx=P, pady=(4, P))
        self._log = tk.Text(fl, height=12, width=74, state='disabled',
                            font=('Consolas', 9), bg='#1e1e1e', fg='#d4d4d4')
        sb = ttk.Scrollbar(fl, command=self._log.yview)
        self._log.config(yscrollcommand=sb.set)
        self._log.grid(row=0, column=0, sticky='nsew')
        sb.grid(row=0, column=1, sticky='ns')

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_meta_panel(self, parent):
        """4-tab notebook matching the GMS Image Info metadata sections."""
        P2 = 4

        nb = ttk.Notebook(parent)
        nb.pack(fill='both', expand=True, padx=2, pady=2)

        def _row(tab, r, label, var, width=12, col_offset=0):
            ttk.Label(tab, text=label, anchor='w').grid(
                row=r, column=col_offset + 0, sticky='w',
                padx=(P2, 2), pady=2)
            ttk.Entry(tab, textvariable=var, width=width).grid(
                row=r, column=col_offset + 1, sticky='w',
                padx=(2, P2 + 10), pady=2)

        # ── Tab 1: Microscope Info ─────────────────────────────────────────────
        t1 = ttk.Frame(nb)
        nb.add(t1, text='Microscope Info')
        mic_fields = [
            ('Beam Energy (kV)',       'beam_energy_kv',      10),
            ('Magnification',          'magnification',       10),
            ('Camera Length (mm)',     'camera_length_mm',    10),
            ('Cs (mm)',                'cs_mm',               10),
            ('Emission Current (uA)',  'emission_current_ua', 10),
            ('Probe Current (nA)',     'probe_current_na',    10),
            ('Probe Size (nm)',        'probe_size_nm',       10),
            ('Mode',                   'mode',                22),
        ]
        # Two columns of 4 rows each to keep the tab compact
        for i, (lbl, key, w) in enumerate(mic_fields):
            col = (i // 4) * 2        # 0 for rows 0-3, 2 for rows 4-7
            row = i % 4
            _row(t1, row, lbl, self._meta[key], width=w, col_offset=col)

        # ── Tab 2: Session Info ────────────────────────────────────────────────
        t2 = ttk.Frame(nb)
        nb.add(t2, text='Session Info')
        for r, (lbl, key) in enumerate([
            ('Specimen',   'specimen'),
            ('Operator',   'operator'),
            ('Microscope', 'microscope'),
            ('Custom',     'custom'),
        ]):
            _row(t2, r, lbl, self._meta[key], width=30)

        # ── Tab 3: EELS ───────────────────────────────────────────────────────
        t3 = ttk.Frame(nb)
        nb.add(t3, text='EELS')
        for r, (lbl, key) in enumerate([
            ('Convergence semi-angle (mrad)', 'convergence_angle_mrad'),
            ('Collection semi-angle (mrad)',  'collection_angle_mrad'),
            ('Entrance Aperture (mm)',        'entrance_aperture_mm'),
            ('Exposure (s)',                  'exposure_s'),
            ('Frames Summed',                 'frames'),
            ('Integration Time (s)',          'integration_time_s'),
        ]):
            _row(t3, r, lbl, self._meta[key], width=12)

        # ── Tab 4: STEM SI ────────────────────────────────────────────────────
        t4 = ttk.Frame(nb)
        nb.add(t4, text='STEM SI')
        for r, (lbl, key) in enumerate([
            ('Pixel Time (s)', 'pixel_time_s'),
            ('Scan Device',    'scan_device'),
        ]):
            _row(t4, r, lbl, self._meta[key], width=24)

    def _sel(self, fmt):
        self._fmt.set(fmt)
        self._btn_dm3.config(
            bg='#2d7fd4' if fmt == 'dm3' else '#cde5fa',
            relief='sunken' if fmt == 'dm3' else 'raised')
        self._btn_raw.config(
            bg='#c17f24' if fmt == 'dat' else '#fde8c0',
            relief='sunken' if fmt == 'dat' else 'raised')
        self._refresh_out()

    def _on_mode_changed(self):
        batch = (self._mode.get() == 'batch')
        self._fi_frame.configure(text='Input Folder' if batch else 'Input File  (.nhdf)')
        # toggle single-mode output widgets
        for w in (self._lbl_path, self._ent_path, self._btn_save_as,
                  self._lbl_file, self._ent_file):
            if batch: w.grid_remove()
            else:     w.grid()
        # toggle batch-mode output widgets
        if batch:
            self._lbl_out_folder.grid(       row=0, column=0, sticky='w',  padx=(0, 4))
            self._ent_out_folder.grid(       row=0, column=1, sticky='ew', padx=(0, 6))
            self._btn_browse_out_folder.grid(row=0, column=2)
        else:
            self._lbl_out_folder.grid_remove()
            self._ent_out_folder.grid_remove()
            self._btn_browse_out_folder.grid_remove()
        self._refresh_out()

    def _on_in_changed(self, *_):
        self._refresh_out()

    def _refresh_out(self):
        inp = self._in_var.get().strip()
        if not inp:
            return
        _ds = datetime.date.today().strftime('%Y%m%d') + 'o'
        if self._mode.get() == 'single':
            fmt   = self._fmt.get()
            _full = os.path.splitext(inp)[0] + '_' + _ds + '.' + fmt
            self._out_dir_var.set(os.path.dirname(_full))
            self._out_name_var.set(os.path.basename(_full))
        else:
            # batch: output folder defaults to the input folder
            self._out_folder_var.set(inp)

    def _browse_in(self):
        if self._mode.get() == 'batch':
            p = filedialog.askdirectory(title='Select input folder')
        else:
            p = filedialog.askopenfilename(
                title='Select NHDF file',
                filetypes=[('Nion Swift', '*.nhdf'), ('HDF5', '*.h5 *.hdf5'), ('All', '*.*')])
        if p:
            self._in_var.set(p)   # trace fires _refresh_out automatically

    def _browse_out(self):
        fmt = self._fmt.get()
        ft  = ([('Gatan DM3', '*.dm3'), ('All', '*.*')] if fmt == 'dm3'
               else [('Raw binary', '*.dat'), ('All', '*.*')])
        p = filedialog.asksaveasfilename(
            title='Save output as',
            defaultextension=('.' + fmt),
            filetypes=ft)
        if p:
            self._out_dir_var.set(os.path.dirname(p))
            self._out_name_var.set(os.path.basename(p))

    def _browse_out_folder(self):
        p = filedialog.askdirectory(title='Select output folder')
        if p:
            self._out_folder_var.set(p)

    def _log_msg(self, msg):
        self._log.config(state='normal')
        self._log.insert('end', msg + '\n')
        self._log.see('end')
        self._log.config(state='disabled')
        self.update_idletasks()

    # ── conversion ────────────────────────────────────────────────────────────

    def _collect_meta(self):
        meta = {}
        for k, var in self._meta.items():
            try:    meta[k] = var.get()
            except: meta[k] = ''
        return meta

    def _convert_one(self, in_path, out_path, fmt, meta):
        """Convert a single NHDF file to out_path.  Logs progress via _log_msg."""
        self._log_msg(f'Reading  {os.path.basename(in_path)} …')
        data, props, cal_x, cal_y, cal_e, title = read_nhdf(in_path)
        ny, nx, ne = data.shape

        self._log_msg(f'  Shape    : ({ny}, {nx}, {ne})   ny × nx × n_energy')
        self._log_msg(f'  Title    : {title}')
        self._log_msg(f'  dtype    : {data.dtype}   ({data.itemsize} bytes/value)')
        self._log_msg(f'  X cal    : offset={cal_x[0]:.6g} {cal_x[2]}'
                      f',  scale={cal_x[1]:.6g} {cal_x[2]}/px')
        self._log_msg(f'  Y cal    : offset={cal_y[0]:.6g} {cal_y[2]}'
                      f',  scale={cal_y[1]:.6g} {cal_y[2]}/px')
        self._log_msg(f'  E cal    : offset={cal_e[0]:.6g} {cal_e[2]}'
                      f',  scale={cal_e[1]:.6g} {cal_e[2]}/ch')
        _e_zlp = (round(-cal_e[0] / cal_e[1]) if cal_e[1] != 0 else 0)
        self._log_msg(f'  E range  : ch 0 → ch {ne-1}   (ZLP @ ch {_e_zlp})')
        self._log_msg(f'  Data     : min={data.min():.4g}'
                      f'  max={data.max():.4g}  mean={data.mean():.4g}')

        if fmt == 'dm3':
            self._log_msg('\nBuilding DM3 from embedded template ...')
            _ic = props.get('intensity_calibration', {})
            cal_i_dm3 = (float(_ic.get('offset', 0.0)),
                         float(_ic.get('scale',  1.0)),
                         str(_ic.get('units', '')))
            out = build_dm3_from_nhdf(data, cal_x=cal_x, cal_y=cal_y,
                                      cal_e=cal_e, cal_i=cal_i_dm3, title=title)
            self._log_msg(f'  DM3 size     : {len(out):,} bytes'
                          f'  ({len(out)/1024/1024:.3f} MB)')
            self._log_msg(f'  Data payload : {data.nbytes:,} bytes'
                          f'  ({data.nbytes*100/len(out):.1f}% of file)')
            self._log_msg(f'\nWriting  {os.path.basename(out_path)} …')
            with open(out_path, 'wb') as fh:
                fh.write(out); fh.flush(); os.fsync(fh.fileno())
            written_size = os.path.getsize(out_path)
            if written_size != len(out):
                self._log_msg(f'  ✗ SIZE MISMATCH: {written_size:,} B written,'
                              f' expected {len(out):,} B')
            else:
                self._log_msg(f'  File size   : {written_size:,} bytes ✓')
            with open(out_path, 'rb') as fh:
                blob = fh.read()
            needle = data.transpose(2, 0, 1).tobytes()[:32]
            off = blob.find(needle)
            if off >= 0:
                check    = np.frombuffer(blob[off:off+32], dtype='<f4')
                expected = data.transpose(2, 0, 1).ravel()[:8]
                ok = np.allclose(check, expected, rtol=0, atol=0)
                self._log_msg(f'  SI data @ offset {off:,}  ✓  '
                              + ('first 8 values match ✓' if ok else 'WARNING: values differ'))
            else:
                self._log_msg('  WARNING: SI data block not found')
            self._log_msg(f'\n  → {out_path}')

        else:
            self._log_msg('\nWriting DAT …')
            _ic = props.get('intensity_calibration', {})
            cal_i = (float(_ic.get('offset', 0.0)),
                     float(_ic.get('scale',  1.0)),
                     str(_ic.get('units', '')))
            nbytes, txt_path = write_raw(out_path, data,
                                         cal_x=cal_x, cal_y=cal_y,
                                         cal_e=cal_e, cal_i=cal_i, meta=meta)
            self._log_msg(f'  Written  : {nbytes:,} bytes  ({nbytes/1024/1024:.3f} MB)')
            self._log_msg(f'  Values   : {data.size:,} float32  (4 bytes each)')
            self._log_msg( '  Layout   : C-order  (ne slowest, nx fastest)')
            self._log_msg(f'  Shape    : ({ne}, {ny}, {nx})  = ne × ny × nx')
            rd   = np.fromfile(out_path, dtype='<f4').reshape(ne, ny, nx)
            diff = float(np.max(np.abs(rd - data.transpose(2, 0, 1))))
            ok   = diff < 1e-9
            self._log_msg(f'  Round-trip max delta = {diff:.2e}  '
                          + ('✓ PASS' if ok else '✗ FAIL'))
            _nnd = sum(1 for v in meta.values() if str(v) not in ('', '0', '0.0', '1'))
            self._log_msg(f'  Metadata : {len(meta)} fields written  ({_nnd} non-default)')
            self._log_msg(f'\n  → {out_path}')
            self._log_msg(f'  → {txt_path}  (sidecar)')

    def _convert(self):
        fmt  = self._fmt.get()
        meta = self._collect_meta()

        # ── build job list ────────────────────────────────────────────────────
        if self._mode.get() == 'single':
            in_path  = self._in_var.get().strip()
            out_path = os.path.join(self._out_dir_var.get().strip(),
                                    self._out_name_var.get().strip())
            if not in_path or not os.path.isfile(in_path):
                messagebox.showerror('Error', 'Please select a valid input file.')
                return
            if not self._out_name_var.get().strip():
                messagebox.showerror('Error', 'Please specify an output file name.')
                return
            jobs = [(in_path, out_path)]
        else:
            in_folder  = self._in_var.get().strip()
            out_folder = self._out_folder_var.get().strip()
            if not in_folder or not os.path.isdir(in_folder):
                messagebox.showerror('Error', 'Please select a valid input folder.')
                return
            if not out_folder or not os.path.isdir(out_folder):
                messagebox.showerror('Error', 'Please select a valid output folder.')
                return
            _ds = datetime.date.today().strftime('%Y%m%d') + 'o'
            nhdf_files = sorted(f for f in os.listdir(in_folder)
                                if f.lower().endswith('.nhdf'))
            if not nhdf_files:
                messagebox.showerror('Error', 'No .nhdf files found in the input folder.')
                return
            jobs = [(os.path.join(in_folder, f),
                     os.path.join(out_folder,
                                  os.path.splitext(f)[0] + '_' + _ds + '.' + fmt))
                    for f in nhdf_files]

        self._btn_conv.state(['disabled'])
        self._prog.start(12)

        def _run():
            n_ok = 0
            try:
                for i, (in_p, out_p) in enumerate(jobs):
                    if len(jobs) > 1:
                        self._log_msg(f'\n── {i+1}/{len(jobs)}: '
                                      f'{os.path.basename(in_p)} ──')
                    self._convert_one(in_p, out_p, fmt, meta)
                    n_ok += 1
                sep = '─' * 52
                if len(jobs) > 1:
                    self._log_msg(f'\nBatch complete: {n_ok}/{len(jobs)} converted.\n{sep}')
                    messagebox.showinfo('Done', f'{n_ok} file(s) saved to:\n{out_folder}')
                else:
                    self._log_msg(f'\nConversion complete.\n{sep}')
                    messagebox.showinfo('Done', f'Saved to:\n{jobs[0][1]}')
            except Exception as exc:
                self._log_msg(f'\nERROR: {exc}\n{traceback.format_exc()}')
                messagebox.showerror('Failed', str(exc))
            finally:
                self._prog.stop()
                self._btn_conv.state(['!disabled'])

        threading.Thread(target=_run, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    App().mainloop()
