#include"kernels.cuh"

/* ### device functions ### */
/* ## geometry functions ## */
__device__ bool inside_triangle(Triangle t, Point p)
{
    double area = 0.5 * (-t.b.y * t.c.x + t.a.y * (-t.b.x + t.c.x) + t.a.x * (t.b.y - t.c.y) + t.b.x * t.c.y);

    int sign = 1;
    if(area<0)
        sign = -1;

    double s = (t.a.y * t.c.x - t.a.x * t.c.y + (t.c.y - t.a.y) * p.x + (t.a.x - t.c.x) * p.y) * sign;
    double t = (t.a.x * t.b.y - t.a.y * t.b.x + (t.a.y - t.b.y) * p.x + (t.b.x - t.a.x) * p.y) * sign;
    
    return (s > 0) && (t > 0) && ((s + t) < 2 * A * sign);
}

__device__ bool inCircle(Triangle t, Point p)
{
    //TODO
}
/* ------------------------------------------------------------------------------------- */

/* ## data structor functions ## */
/* #     suspect data sets     # */
__device__ SuspectSet makeSuspectSet()
{
    //TODO
}

__device__ void addSuspect(SuspectSet set, Triangle t, Point edge_a, Point edge_b)
{
    //TODO
}

__device__ Suspect popSuspect(SuspectSet set)
{
    //TODO
}

__device__ bool isNeighbour_suspect(Suspect suspect, Triangle t)
{
    //TODO
}

/* # triangle data set # */
__device__ Triangle makeTriangle(Point a, point b, point c)
{
    //TODO
}

__device__ bool deleteTriangle(Triangle t)
{
    //TODO
}

__device__ Point nextPointTriangle(Triangle t, Point p)
{
    //TODO
}

__device__ Point prePointTriangle(Triangle t, Point p)
{
    //TODO
}
/* ------------------------------------------------------------------------------------- */

__global__ void add_point_incrimental(Regins regins)
{
    /* extract information from data structure and define variables */
    TriangleSet t_set = regins[blockIdx.y][blockIdx.x].triangles;
    PointSet p_set = regins[blockIdx.y][blockIdx.x].points;
    Point p = t_set.pointToAdd;
    int tid = threadIdx.x;

    /* first thread initial susspects list */
    SuspectSet susspects;
    if(tid == 0)
        suspects = makeSuspectSet();

    __syncthreads();

    /* each thread represent a triangle                   */
    /* every triangle check if new point is inside or not */
    if(tid < t_set.size)
    {
        Triangle t = t_set.triangles[tid];
        bool isInside = inside_triangle(t, p);

        /* thread of choosen triangle make new triangles */
        /* and delete its triangle and add suspects      */
        if(isInside)
        {
            Triangle t1 = makeTriangle(p, t.a, t.b);
            Triangle t2 = makeTriangle(p, t.b, t.c);
            Triangle t3 = makeTriangle(p, t.c, t.a);

            deleteTriangle(t);

            addSuspect(suspects, t1, t1.b, t1.c);
            addSuspect(suspects, t2, t2.b, t2.c);
            addSuspect(suspects, t3, t3.b, t3.c);
        }
    }

    /* each thread represent a point                                     */
    /* for each suspect every point(thread) checks for delouney triangle */
    if(tid < p_set.size)
    {
        while(suspects.size != 0)
        {
            Suspect current;
            if(tid == 0) current = popSuspect(suspects);
            bool fail = false;
    
            __syncthreads();
    
            if(inCircle(current.triangle, p_set[tid]))
            {
                fail = true;    //suspect fail to be a delouney triangle
            }

            __syncthreads();

            /* each thread represent a triangle                                               */
            /* each thread checks that coresponding triangle is neighbour of current triangle */
            /* only the triangle with common suspected edge is included                       */
            if(fail && tid < t_set.size)
            {
                Triangle condidate_neighbour = t_set.triangles[tid];
                if(isNeighbour_suspect(current, condidate_neighbour))
                {
                    /* find far point (third point of neighbour triangle that isn't on common edge) */
                    Point far_point = condidate_neighbour.a;
                    while(true)
                    {
                        if(!isSamePoint(current.triangle.a, far_point) && !isSamePoint(current.triangle.b, far_point)
                            && !isSamePoint(current.triangle.c, far_point))
                            {break ;}
                        far_point = condidate_neighbour.b;
                        if(!isSamePoint(current.triangle.a, far_point) && !isSamePoint(current.triangle.b, far_point)
                            && !isSamePoint(current.triangle.c, far_point))
                            {break ;}
                        far_point = condidate_neighbour.c;
                        break;
                    }

                    /* counter-clock rotation of points in triangle */
                    Point next_p = nextPointTriangle(current.triangle, p);
                    Point pre_p = prePointTriangle(current.triangle, p);

                    /* flip common edge between two triangle */
                    Triangle flip1 = makeTriangle(p, next_p, far_point);
                    Triangle flip2 = makeTriangle(p, far_point, pre_p);

                    deleteTriangle(current.triangle);
                    deleteTriangle(condidate_neighbour);

                    addSuspect(suspects, flip1, next_p, far_point);
                    addSuspect(suspects, flip2, far_point, pre_p);
                }
            }
        }
    }
}

// __global__ void merge(PointSet set, EdgeSet edges)
// {
//     int tid = threadIdx.x;
// }


// __global__ void delauney(Set set)
// {
//     Regin regin = set[threadIdx.y][threadIdx.x];
// }