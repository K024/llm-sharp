import re
from pathlib import Path

root = Path(__file__).parent


def _hack_file(file: Path, old_text: str, new_text: str):
    file_text = file.read_text()

    if isinstance(old_text, re.Pattern):
        file_text_hacked = re.sub(old_text, new_text, file_text)
    else:
        file_text_hacked = file_text.replace(old_text, new_text)

    file.write_text(file_text_hacked)

    def _restore():
        file.write_text(file_text)

    return _restore


def hack_sources():

    _restores = []

    # disable flash_attention for sm_75 since
    # torch_ext load cannot set the compute capability for each file
    _restores.append(_hack_file(
        root / "third-party/lmdeploy/src/turbomind/models/llama/llama_kernels.cu",
        "constexpr static int CONST_NAME = 2;",
        "constexpr static int CONST_NAME = 1;",
    ))

    # hack for cuda 11.7 with nvcc 12
    _restores.append(_hack_file(
        root / "third-party/cutlass/include/cutlass/cutlass.h",
        "#pragma once",
        "#pragma once\n\n"
        "#define __CUDACC_VER_MAJOR__ 11\n"
        "#define __CUDACC_VER_MINOR__ 7\n",
    ))

    def restore():
        for r in _restores:
            r()

    return restore
