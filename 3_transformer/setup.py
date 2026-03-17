from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

setup_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='naive-cu-extensions',
    ext_modules=[
        CUDAExtension(
            name='custom_training_extension',
            sources=[
                os.path.join(setup_dir, 'csrc', 'training', 'binding.cpp'),
                os.path.join(setup_dir, 'csrc', 'training', 'kernels', 'matmul.cu'),
                os.path.join(setup_dir, 'csrc', 'training', 'kernels', 'elementwise.cu'),
                os.path.join(setup_dir, 'csrc', 'training', 'kernels', 'activation.cu'),
                os.path.join(setup_dir, 'csrc', 'training', 'kernels', 'softmax.cu'),
                os.path.join(setup_dir, 'csrc', 'training', 'kernels', 'layernorm.cu'),
                os.path.join(setup_dir, 'csrc', 'training', 'kernels', 'embedding.cu'),
            ],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-O2']
            }
        ),
        CUDAExtension(
            name='custom_inference_extension',
            sources=[
                os.path.join(setup_dir, 'csrc', 'inference', 'binding.cpp'),
                os.path.join(setup_dir, 'csrc', 'inference', 'kernels', 'matmul_fwd.cu'),
                os.path.join(setup_dir, 'csrc', 'inference', 'kernels', 'gemv_fwd.cu'),
                os.path.join(setup_dir, 'csrc', 'inference', 'kernels', 'elementwise_fwd.cu'),
                os.path.join(setup_dir, 'csrc', 'inference', 'kernels', 'activation_fwd.cu'),
                os.path.join(setup_dir, 'csrc', 'inference', 'kernels', 'softmax_fwd.cu'),
                os.path.join(setup_dir, 'csrc', 'inference', 'kernels', 'layernorm_fwd.cu'),
                os.path.join(setup_dir, 'csrc', 'inference', 'kernels', 'topk_fwd.cu'),
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--expt-extended-lambda']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
