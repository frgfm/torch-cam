Installation
============

In this tutorial, you will need the entire project codebase. So first,
we clone the project’s GitHub repository and install from source.

.. code-block::python

    >>> !git clone https://github.com/frgfm/torch-cam.git
    >>> !pip install -e torch-cam/.


Hardware information
====================

GPU information
---------------

To be able to run the benchmark on GPU, you need to have the correct
driver and CUDA installation. If you get a message starting with: >
NVIDIA-SMI has failed…

The script will be running on CPU as PyTorch isn’t able to access any
CUDA-capable device.

.. code-block::python

    >>> !nvidia-smi
    NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.


CPU information
---------------

Latency will vary greatly depending on the capabilities of your CPU.
Some models are optimized for CPU architectures (MobileNet V3 for
instance), while others were only designed for GPU and will thus yield
poor latency when run on CPU.

.. code-block::python

    >>> !lscpu
    Architecture:        x86_64
    CPU op-mode(s):      32-bit, 64-bit
    Byte Order:          Little Endian
    CPU(s):              2
    On-line CPU(s) list: 0,1
    Thread(s) per core:  2
    Core(s) per socket:  1
    Socket(s):           1
    NUMA node(s):        1
    Vendor ID:           AuthenticAMD
    CPU family:          23
    Model:               49
    Model name:          AMD EPYC 7B12
    Stepping:            0
    CPU MHz:             2249.998
    BogoMIPS:            4499.99
    Hypervisor vendor:   KVM
    Virtualization type: full
    L1d cache:           32K
    L1i cache:           32K
    L2 cache:            512K
    L3 cache:            16384K
    NUMA node0 CPU(s):   0,1
    Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc cpuid extd_apicid tsc_known_freq pni pclmulqdq ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext ssbd ibrs ibpb stibp vmmcall fsgsbase tsc_adjust bmi1 avx2 smep bmi2 rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 clzero xsaveerptr arat npt nrip_save umip rdpid


Usage
=====

The latency evaluation script provides several options for you to play
with: change the input size, the architecture or the CAM method to
better reflect your use case.

.. code-block::python

    >>> !cd torch-cam/ && python scripts/eval_latency.py --help
    usage: eval_latency.py [-h] [--arch ARCH] [--size SIZE]
                           [--class-idx CLASS_IDX] [--device DEVICE] [--it IT]
                           method

    CAM method latency benchmark

    positional arguments:
      method                CAM method to use

    optional arguments:
      -h, --help            show this help message and exit
      --arch ARCH           Name of the torchvision architecture (default:
                            resnet18)
      --size SIZE           The image input size (default: 224)
      --class-idx CLASS_IDX
                            Index of the class to inspect (default: 232)
      --device DEVICE       Default device to perform computation on (default:
                            None)
      --it IT               Number of iterations to run (default: 100)


Architecture designed for GPU
-----------------------------

Let’s benchmark the latency of CAM methods with the popular ResNet
architecture

.. code-block::python

    >>> !cd torch-cam/ && python scripts/eval_latency.py SmoothGradCAMpp --arch resnet18
    Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
    100% 44.7M/44.7M [00:00<00:00, 85.9MB/s]
    /usr/local/lib/python3.7/dist-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.)
      return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
    WARNING:root:no value was provided for `target_layer`, thus set to 'layer4'.
    SmoothGradCAMpp w/ resnet18 (100 runs on (224, 224) inputs)
    mean 1143.17ms, std 36.79ms


.. code-block::python

    >>> !cd torch-cam/ && python scripts/eval_latency.py LayerCAM --arch resnet18
    /usr/local/lib/python3.7/dist-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.)
      return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
    WARNING:root:no value was provided for `target_layer`, thus set to 'layer4'.
    LayerCAM w/ resnet18 (100 runs on (224, 224) inputs)
    mean 189.64ms, std 8.82ms


Architecture designed for CPU
-----------------------------

As mentioned, we’ll consider MobileNet V3 here.

.. code-block::python

    >>> !cd torch-cam/ && python scripts/eval_latency.py SmoothGradCAMpp --arch mobilenet_v3_large
    Downloading: "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth" to /root/.cache/torch/hub/checkpoints/mobilenet_v3_large-8738ca79.pth
    100% 21.1M/21.1M [00:00<00:00, 71.5MB/s]
    WARNING:root:no value was provided for `target_layer`, thus set to 'features.4.block.1'.
    SmoothGradCAMpp w/ mobilenet_v3_large (100 runs on (224, 224) inputs)
    mean 762.18ms, std 26.95ms


.. code-block::python

    >>> !cd torch-cam/ && python scripts/eval_latency.py LayerCAM --arch mobilenet_v3_large
    WARNING:root:no value was provided for `target_layer`, thus set to 'features.4.block.1'.
    LayerCAM w/ mobilenet_v3_large (100 runs on (224, 224) inputs)
    mean 148.76ms, std 7.86ms
