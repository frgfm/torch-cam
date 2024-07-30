# Copyright (C) 2020-2023, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""
Based on https://github.com/pytorch/pytorch/blob/master/torch/utils/collect_env.py
This script outputs relevant system environment info
Run it with `python collect_env.py`.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import locale
import os
import re
import subprocess  # noqa S404
import sys
from pathlib import Path
from typing import NamedTuple

try:
    import torchcam

    TORCHCAM_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    TORCHCAM_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    TORCH_AVAILABLE = False

PY3 = sys.version_info >= (3, 0)


# System Environment Information
class SystemEnv(NamedTuple):
    torchcam_version: str
    torch_version: str
    os: str
    python_version: str
    is_cuda_available: bool
    cuda_runtime_version: str
    nvidia_driver_version: str
    nvidia_gpu_models: str
    cudnn_version: str


def run(command):
    """Returns (return-code, stdout, stderr)"""
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, err = p.communicate()
    rc = p.returncode
    if PY3:
        enc = locale.getpreferredencoding()
        output = output.decode(enc)
        err = err.decode(enc)
    return rc, output.strip(), err.strip()


def run_and_read_all(run_lambda, command):
    """Runs command using run_lambda; reads and returns entire output if rc is 0"""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    return out


def run_and_parse_first_match(run_lambda, command, regex):
    """Runs command using run_lambda, returns the first regex match if it exists"""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    match = re.search(regex, out)
    if match is None:
        return None
    return match.group(1)


def get_nvidia_driver_version(run_lambda):
    if get_platform() == "darwin":
        cmd = "kextstat | grep -i cuda"
        return run_and_parse_first_match(run_lambda, cmd, r"com[.]nvidia[.]CUDA [(](.*?)[)]")
    smi = get_nvidia_smi()
    return run_and_parse_first_match(run_lambda, smi, r"Driver Version: (.*?) ")


def get_gpu_info(run_lambda):
    if get_platform() == "darwin":
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return torch.cuda.get_device_name(None)
        return None
    smi = get_nvidia_smi()
    uuid_regex = re.compile(r" \(UUID: .+?\)")
    rc, out, _ = run_lambda(smi + " -L")
    if rc != 0:
        return None
    # Anonymize GPUs by removing their UUID
    return re.sub(uuid_regex, "", out)


def get_running_cuda_version(run_lambda):
    return run_and_parse_first_match(run_lambda, "nvcc --version", r"release .+ V(.*)")


def get_cudnn_version(run_lambda):
    """This will return a list of libcudnn.so; it's hard to tell which one is being used"""
    if get_platform() == "win32":
        cudnn_cmd = 'where /R "%CUDA_PATH%\\bin" cudnn*.dll'
    elif get_platform() == "darwin":
        # CUDA libraries and drivers can be found in /usr/local/cuda/. See
        # https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html#install
        # https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installmac
        # Use CUDNN_LIBRARY when cudnn library is installed elsewhere.
        cudnn_cmd = "ls /usr/local/cuda/lib/libcudnn*"
    else:
        cudnn_cmd = 'ldconfig -p | grep libcudnn | rev | cut -d" " -f1 | rev'
    rc, out, _ = run_lambda(cudnn_cmd)
    # find will return 1 if there are permission errors or if not found
    if len(out) == 0 or rc not in (1, 0):
        lib = os.environ.get("CUDNN_LIBRARY")
        if lib is not None and Path(lib).is_file():
            return os.path.realpath(lib)
        return None
    files = set()
    for fn in out.split("\n"):
        fn = os.path.realpath(fn)  # eliminate symbolic links
        if Path(fn).is_file():
            files.add(fn)
    if not files:
        return None
    # Alphabetize the result because the order is non-deterministic otherwise
    files = sorted(files)
    if len(files) == 1:
        return files[0]
    result = "\n".join(files)
    return "Probably one of the following:\n{}".format(result)


def get_nvidia_smi():
    # Note: nvidia-smi is currently available only on Windows and Linux
    smi = "nvidia-smi"
    if get_platform() == "win32":
        system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")
        program_files_root = os.environ.get("PROGRAMFILES", "C:\\Program Files")
        legacy_path = Path(program_files_root) / "NVIDIA Corporation" / "NVSMI" / smi
        new_path = Path(system_root) / "System32" / smi
        smis = [new_path, legacy_path]
        for candidate_smi in smis:
            if Path(candidate_smi).exists():
                smi = '"{}"'.format(candidate_smi)
                break
    return smi


def get_platform():
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform.startswith("win32"):
        return "win32"
    if sys.platform.startswith("cygwin"):
        return "cygwin"
    if sys.platform.startswith("darwin"):
        return "darwin"
    return sys.platform


def get_mac_version(run_lambda):
    return run_and_parse_first_match(run_lambda, "sw_vers -productVersion", r"(.*)")


def get_windows_version(run_lambda):
    return run_and_read_all(run_lambda, "wmic os get Caption | findstr /v Caption")


def get_lsb_version(run_lambda):
    return run_and_parse_first_match(run_lambda, "lsb_release -a", r"Description:\t(.*)")


def check_release_file(run_lambda):
    return run_and_parse_first_match(run_lambda, "cat /etc/*-release", r'PRETTY_NAME="(.*)"')


def get_os(run_lambda):
    platform = get_platform()

    if platform in ("win32", "cygwin"):
        return get_windows_version(run_lambda)

    if platform == "darwin":
        version = get_mac_version(run_lambda)
        if version is None:
            return None
        return "Mac OSX {}".format(version)

    if platform == "linux":
        # Ubuntu/Debian based
        desc = get_lsb_version(run_lambda)
        if desc is not None:
            return desc

        # Try reading /etc/*-release
        desc = check_release_file(run_lambda)
        if desc is not None:
            return desc

        return platform

    # Unknown platform
    return platform


def get_env_info():
    run_lambda = run

    torchcam_str = torchcam.__version__ if TORCHCAM_AVAILABLE else "N/A"

    if TORCH_AVAILABLE:
        torch_str = torch.__version__
        cuda_available_str = torch.cuda.is_available()
    else:
        torch_str = cuda_available_str = "N/A"

    return SystemEnv(
        torchcam_version=torchcam_str,
        torch_version=torch_str,
        python_version=".".join(map(str, sys.version_info[:3])),
        is_cuda_available=cuda_available_str,
        cuda_runtime_version=get_running_cuda_version(run_lambda),
        nvidia_gpu_models=get_gpu_info(run_lambda),
        nvidia_driver_version=get_nvidia_driver_version(run_lambda),
        cudnn_version=get_cudnn_version(run_lambda),
        os=get_os(run_lambda),
    )


env_info_fmt = """
TorchCAM version: {torchcam_version}
PyTorch version: {torch_version}

OS: {os}

Python version: {python_version}
Is CUDA available: {is_cuda_available}
CUDA runtime version: {cuda_runtime_version}
GPU models and configuration: {nvidia_gpu_models}
Nvidia driver version: {nvidia_driver_version}
cuDNN version: {cudnn_version}
""".strip()


def pretty_str(envinfo):
    def replace_nones(dct, replacement="Could not collect"):
        for key in dct:
            if dct[key] is not None:
                continue
            dct[key] = replacement
        return dct

    def replace_bools(dct, true="Yes", false="No"):
        for key in dct:
            if dct[key] is True:
                dct[key] = true
            elif dct[key] is False:
                dct[key] = false
        return dct

    def maybe_start_on_next_line(string):
        # If `string` is multiline, prepend a \n to it.
        if string is not None and len(string.split("\n")) > 1:
            return "\n{}\n".format(string)
        return string

    mutable_dict = envinfo._asdict()

    # If nvidia_gpu_models is multiline, start on the next line
    mutable_dict["nvidia_gpu_models"] = maybe_start_on_next_line(envinfo.nvidia_gpu_models)

    # If the machine doesn't have CUDA, report some fields as 'No CUDA'
    dynamic_cuda_fields = [
        "cuda_runtime_version",
        "nvidia_gpu_models",
        "nvidia_driver_version",
    ]
    all_cuda_fields = [*dynamic_cuda_fields, "cudnn_version"]
    all_dynamic_cuda_fields_missing = all(mutable_dict[field] is None for field in dynamic_cuda_fields)
    if TORCH_AVAILABLE and not torch.cuda.is_available() and all_dynamic_cuda_fields_missing:
        for field in all_cuda_fields:
            mutable_dict[field] = "No CUDA"

    # Replace True with Yes, False with No
    mutable_dict = replace_bools(mutable_dict)

    # Replace all None objects with 'Could not collect'
    mutable_dict = replace_nones(mutable_dict)

    return env_info_fmt.format(**mutable_dict)


def get_pretty_env_info():
    """Collects environment information for debugging purposes

    Returns:
        str: environment information
    """
    return pretty_str(get_env_info())


def main():
    print("Collecting environment information...")
    output = get_pretty_env_info()
    print(output)


if __name__ == "__main__":
    main()
