#include"kernels.cuh"

__device__ Edge find_LR_base(PointSet set)
{

}

__global__ void merge(PointSet set, EdgeSet edges)
{
    int tid = threadIdx.x;
}

__global__ void delauney(Set set)
{
    Regin regin = set[threadIdx.y][threadIdx.x];
}