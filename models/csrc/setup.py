from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_ext_modules():
    return [
        CUDAExtension(
            name='_msmv_sampling_cuda',
            sources=[
                'msmv_sampling/msmv_sampling.cpp',
                'msmv_sampling/msmv_sampling_forward.cu',
                'msmv_sampling/msmv_sampling_backward.cu'
            ],
            include_dirs=['msmv_sampling'],
            extra_compile_args=dict(
                nvcc=[
                    "-gencode=arch=compute_60,code=sm_60",
                    "-gencode=arch=compute_61,code=sm_61",
                    "-gencode=arch=compute_70,code=sm_70",
                    "-gencode=arch=compute_75,code=sm_75",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86",
                    "-gencode=arch=compute_86,code=compute_86",
                ]
            )
        ),
        CUDAExtension(
            name='_bev_pool_v2_cuda',
            sources=[
                'bev_pool_v2/bev_pool_v2.cpp',
                'bev_pool_v2/bev_pool_v2_kernel.cu',
            ],
            include_dirs=['bev_pool_v2'],
            extra_compile_args=dict(
                nvcc=[
                    "-gencode=arch=compute_60,code=sm_60",
                    "-gencode=arch=compute_61,code=sm_61",
                    "-gencode=arch=compute_70,code=sm_70",
                    "-gencode=arch=compute_75,code=sm_75",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86",
                    "-gencode=arch=compute_89,code=sm_89",
                    "-gencode=arch=compute_89,code=compute_89",
                ]
            )
        ),
        CUDAExtension(
            name='_local_aggregate_cuda',
            sources=[
                'local_aggregate/local_aggregate.cpp',
                'local_aggregate/local_aggregate_cuda.cu',
                'local_aggregate/aggregator_impl.cu',
                'local_aggregate/forward.cu',
                'local_aggregate/backward.cu',
            ],
            include_dirs=['local_aggregate'],
            extra_compile_args=dict(
                nvcc=["-Xcompiler", "-fno-gnu-unique"]
            )
        ),
        CUDAExtension(
            name='_deformable_aggregation_cuda',
            sources=[
                'deformable_aggregation/deformable_aggregation.cpp',
                'deformable_aggregation/deformable_aggregation_cuda.cu',
            ],
            include_dirs=['deformable_aggregation'],
            extra_compile_args=dict(
                nvcc=[
                    "-D__CUDA_NO_HALF_OPERATORS__",
                    "-D__CUDA_NO_HALF_CONVERSIONS__",
                    "-D__CUDA_NO_HALF2_OPERATORS__",
                ]
            )
        )
    ]


setup(
    name='csrc',
    ext_modules=get_ext_modules(),
    cmdclass={'build_ext': BuildExtension}
)
