#ifndef _DATASETLIB_H_
#define _DATASETLIB_H_

#define  otherPoint(e,p)  (edgeSet.edges[e].origin == p ? edgeSet.edges[e].destination : edgeSet.edges[e].origin)
#define  previousEdge(e,p)  (edgeSet.edges[e].origin == p ? edgeSet.edges[e].originPrev : edgeSet.edges[e].destinationPrev)
#define  nextEdge(e,p)  (edgeSet.edges[e].origin == p ? edgeSet.edges[e].originNext : edgeSet.edges[e].destinationNext)
#define  makeVector(p1,p2,u,v) (u = pointSet.points[p2].x - pointSet.points[p1].x, v = pointSet.points[p2].y - pointSet.points[p1].y)
#define  crossProduct(u1,v1,u2,v2) (u1 * v2 - v1 * u2)
#define  dotProduct(u1,v1,u2,v2) (u1 * u2 + v1 * v2)

#define TRUE 1
#define FALSE 0

typedef struct Point Point;
typedef struct Edge Edge;
typedef unsigned int boolean;

typedef struct Point
{
    double x;
    double y;
    long entryPoint;
} Point;

typedef struct Edge
{
    long origin;
    long destination;

    long originNext;
    long originPrev;
    long destinationNext;
    long destinationPrev;
} Edge;

typedef struct PointSet
{
    Point * points;
    int size;
} PointSet;

typedef struct EdgeSet
{
    Edge * edges;
    int size;
} EdgeSet;

// pointSet genrate_random_dataset();
// pointSet read_dataset_file(const char *);

#endif