#ifndef _KERNELS_CUH_
#define _KERNELS_CUH_

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

#include<cuda.h>
#include"datasetlib.h"

/*  geometry functions  */
__device__ bool inside_triangle(Triangle t, Point p);
__device__ bool inCircle(Triangle t, Point p);
/*  suspect data sets   */
__device__ TriangleSet makeSuspectSet();
__device__ void addSuspect(TriangleSet set, Triangle t, Point edge_a, Point edge_b);
__device__ Suspect popSuspect(SuspectSet set);
__device__ bool isNeighbour_suspect(Suspect suspect, Triangle t);
/*   triangle data set  */
__device__ Triangle makeTriangle(Point a, point b, point c);
__device__ bool deleteTriangle(Triangle t);
__device__ Point nextPointTriangle(Triangle t, Point p);
__device__ Point prePointTriangle(Triangle t, Point p);

/*    global kernels    */
__global__ void add_point_incrimental(Regins regins);

#endif