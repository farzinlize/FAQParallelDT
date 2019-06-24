#include"delaunay.h"

PointSet pointSet;
EdgeSet edgeSet;

//extern Point *points;
// extern Edge *edges;
// extern long edgeCounter;
// extern long nPoints;
// extern long nEdges;

void delaunay(long start, long end, long *leftEdgeIdx, long *rightEdgeIdx)
{
    long edgeOneIdx, edgeTwoIdx, edgeThreeIdx;
    long leftSubLeftEdgeIdx, leftSubRightEdgeIdx, rightSubLeftEdgeIdx, rightSubRightEdgeIdx, lowerCommonTangentIdx;
    long pointCount = end - start + 1;
    if(pointCount == 2) {
        *leftEdgeIdx = *rightEdgeIdx = createEdge(start, end);
    } else if(pointCount == 3) {
            edgeOneIdx = createEdge(start, start+1);
            edgeTwoIdx = createEdge(start+1, end);
            addEdgeToRing(edgeOneIdx, edgeTwoIdx, start+1);
            double direction = triangleCheck(start, start+1, end);

            if(direction > 0.0) {
                edgeThreeIdx = makeTriangle(edgeOneIdx, start, edgeTwoIdx, end, 1);
                *leftEdgeIdx = edgeOneIdx;
                *rightEdgeIdx = edgeTwoIdx;
            } else if(direction < 0.0) {
                edgeThreeIdx = makeTriangle(edgeOneIdx, start, edgeTwoIdx, end, 0);
                *leftEdgeIdx = edgeThreeIdx;
                *rightEdgeIdx = edgeThreeIdx;
            } else {
                *leftEdgeIdx = edgeOneIdx;
                *rightEdgeIdx = edgeTwoIdx;
            }
    } else if(pointCount > 3) {
        int mid = start + ((end - start)/2);

        delaunay(start, mid, &leftSubLeftEdgeIdx, &rightSubLeftEdgeIdx);
        delaunay(mid+1, end, &leftSubRightEdgeIdx, &rightSubRightEdgeIdx);

        merge(rightSubLeftEdgeIdx, mid, leftSubRightEdgeIdx, mid+1, &lowerCommonTangentIdx);
        //COME BACK HERE, if MUTATING IN MERGE

        if(edgeSet.edges[lowerCommonTangentIdx].origin == start) {
            leftSubLeftEdgeIdx = lowerCommonTangentIdx;
        }
        if(edgeSet.edges[lowerCommonTangentIdx].destination == end) {
            rightSubRightEdgeIdx = lowerCommonTangentIdx;
        }

        *leftEdgeIdx = leftSubLeftEdgeIdx;
        *rightEdgeIdx = rightSubRightEdgeIdx;

    }
}

long createEdge(long startPointIdx, long endPointIdx)
{
    //printf("Creating Edge between %ld and %ld\n", startPointIdx, endPointIdx);
    long c = edgeSet.size;
    edgeSet.edges[c].origin = startPointIdx;
    edgeSet.edges[c].destination = endPointIdx;
    
    edgeSet.edges[c].originNext = edgeSet.edges[c].originPrev = edgeSet.edges[c].destinationNext = edgeSet.edges[c].destinationPrev = c;
    
    if(pointSet.points[startPointIdx].entryPoint == -1) {
        pointSet.points[startPointIdx].entryPoint = c;
    }
    if(pointSet.points[endPointIdx].entryPoint == -1) {
        pointSet.points[endPointIdx].entryPoint = c;
    }

    edgeSet.size++;
    //printEdge(c);
    return c;
}

void addEdgeToRing(long edgeOneIdx, long edgeTwoIdx, long commonPointIdx)
{
    long tempEdgeIdx;
    
    if(edgeSet.edges[edgeOneIdx].origin == commonPointIdx) {
        tempEdgeIdx = edgeSet.edges[edgeOneIdx].originNext;
        edgeSet.edges[edgeOneIdx].originNext = edgeTwoIdx;
    } else {
        tempEdgeIdx = edgeSet.edges[edgeOneIdx].destinationNext;
        edgeSet.edges[edgeOneIdx].destinationNext = edgeTwoIdx;
    }
    if(edgeSet.edges[tempEdgeIdx].origin == commonPointIdx) {
        edgeSet.edges[tempEdgeIdx].originPrev = edgeTwoIdx;
    } else {
        edgeSet.edges[tempEdgeIdx].destinationPrev = edgeTwoIdx;
    }
    if(edgeSet.edges[edgeTwoIdx].origin == commonPointIdx) {
        edgeSet.edges[edgeTwoIdx].originNext = tempEdgeIdx;
        edgeSet.edges[edgeTwoIdx].originPrev = edgeOneIdx;
    } else {
        edgeSet.edges[edgeTwoIdx].destinationNext = tempEdgeIdx;
        edgeSet.edges[edgeTwoIdx].destinationPrev = edgeOneIdx;
    }

}

double triangleCheck(long pointOneIdx, long pointTwoIdx, long pointThreeIdx)
{
    return ( (pointSet.points[pointTwoIdx].x - pointSet.points[pointOneIdx].x) * 
    (pointSet.points[pointThreeIdx].y - pointSet.points[pointOneIdx].y) - 
    (pointSet.points[pointTwoIdx].y - pointSet.points[pointOneIdx].y) * 
    (pointSet.points[pointThreeIdx].x - pointSet.points[pointOneIdx].x));

}


long makeTriangle(long edgeOneIdx, long pointOneIdx, long edgeTwoIdx, long pointTwoIdx, int side)
{

    long tempEdgeIdx;
    tempEdgeIdx = createEdge(pointOneIdx, pointTwoIdx);

    if(side == 0) {
        if(edgeSet.edges[edgeOneIdx].origin == pointOneIdx) {
            addEdgeToRing(edgeSet.edges[edgeOneIdx].originPrev, tempEdgeIdx, pointOneIdx);
        } else {
            addEdgeToRing(edgeSet.edges[edgeOneIdx].destinationPrev, tempEdgeIdx, pointOneIdx);
        }
        addEdgeToRing(edgeTwoIdx, tempEdgeIdx, pointTwoIdx);
    } else {
        addEdgeToRing(edgeOneIdx, tempEdgeIdx, pointOneIdx);
        if(edgeSet.edges[edgeTwoIdx].origin == pointTwoIdx) {
            addEdgeToRing(edgeSet.edges[edgeTwoIdx].originPrev, tempEdgeIdx, pointTwoIdx);
        } else {
            addEdgeToRing(edgeSet.edges[edgeTwoIdx].destinationPrev, tempEdgeIdx, pointTwoIdx);
        }
    }

    return tempEdgeIdx;
}

void findLowerTangent(long rightSubLeftEdgeIdx, long s, long leftSubRightEdgeIdx, long u, long *leftLowerEdgeIdx, long *leftLowerOriginIdx, long *rightLowerEdgeIdx, long *rightLowerOriginIdx)
{
    long left, right;
    long leftOriginIdx, leftDestinationIdx, rightOriginIdx, rightDestinationIdx;
    boolean done;

    left = rightSubLeftEdgeIdx;
    right = leftSubRightEdgeIdx;
    leftOriginIdx = s;
    leftDestinationIdx = otherPoint(left, s);
    
    rightOriginIdx = u;
    rightDestinationIdx = otherPoint(right, u);

    done = FALSE;

    while(!done) {
        if(triangleCheck(leftOriginIdx, leftDestinationIdx, rightOriginIdx) > 0.0) {
            left = previousEdge(left, leftDestinationIdx);
            leftOriginIdx = leftDestinationIdx;
            leftDestinationIdx = otherPoint(left, leftOriginIdx);
        } else if(triangleCheck(rightOriginIdx, rightDestinationIdx, leftOriginIdx) < 0.0) {
            right = nextEdge(right, rightDestinationIdx);
            rightOriginIdx = rightDestinationIdx;
            rightDestinationIdx = otherPoint(right, rightOriginIdx);
        } else {
            done = TRUE;
        }
    }

    *leftLowerEdgeIdx = left;
    *rightLowerEdgeIdx = right;
    *leftLowerOriginIdx = leftOriginIdx;
    *rightLowerOriginIdx = rightOriginIdx;
}

void merge(long rightSubLeftEdgeIdx, long s, long leftSubRightEdgeIdx, long u, long *lowerCommonTangentIdx)
{
    long base, leftCandidateIdx, rightCandidateIdx;
    long baseOriginIdx, baseDestinationIdx;
    double pLeftCandidateOB, qLeftCandidateOB, pLeftCandidateDB, qLeftCandidateDB;
    double pRightCandidateOB, qRightCandidateOB, pRightCandidateDB, qRightCandidateDB;
    double cpLeftCandidate, cpRightCandidate;
    double dpLeftCandidate, dpRightCandidate;

    boolean aboveLeftCandidate, aboveRightCandidate, aboveNext, abovePrev;
    long leftCandidateDestinationIdx, rightCandidateDestinationIdx;
    double cotLeftCandidate, cotRightCandidate;
    long leftLowerEdgeIdx, rightLowerEdgeIdx;
    long rightLowerOriginIdx, leftLowerOriginIdx;

    findLowerTangent(rightSubLeftEdgeIdx, s, leftSubRightEdgeIdx, u, &leftLowerEdgeIdx, &leftLowerOriginIdx, &rightLowerEdgeIdx, &rightLowerOriginIdx);
    base = makeTriangle(leftLowerEdgeIdx, leftLowerOriginIdx, rightLowerEdgeIdx, rightLowerOriginIdx, 1);
    baseOriginIdx = leftLowerOriginIdx;
    baseDestinationIdx = rightLowerOriginIdx;

    *lowerCommonTangentIdx = base;

    do {
        
        leftCandidateIdx = nextEdge(base, baseOriginIdx);
        rightCandidateIdx = previousEdge(base, baseDestinationIdx);

        leftCandidateDestinationIdx = otherPoint(leftCandidateIdx, baseOriginIdx);
        rightCandidateDestinationIdx = otherPoint(rightCandidateIdx, baseDestinationIdx);
        
        makeVector(leftCandidateDestinationIdx, baseOriginIdx, pLeftCandidateOB, qLeftCandidateOB);
        makeVector(leftCandidateDestinationIdx, baseDestinationIdx, pLeftCandidateDB, qLeftCandidateDB);
        makeVector(rightCandidateDestinationIdx, baseOriginIdx, pRightCandidateOB, qRightCandidateOB);
        makeVector(rightCandidateDestinationIdx, baseDestinationIdx, pRightCandidateDB, qRightCandidateDB);

        cpLeftCandidate = crossProduct(pLeftCandidateOB, qLeftCandidateOB, pLeftCandidateDB, qLeftCandidateDB);
        cpRightCandidate = crossProduct(pRightCandidateOB, qRightCandidateOB, pRightCandidateDB, qRightCandidateDB);
        
        aboveLeftCandidate = cpLeftCandidate > 0.0;
        aboveRightCandidate = cpRightCandidate > 0.0;

        if(!aboveLeftCandidate && !aboveRightCandidate) {
            break;
        }        

        if(aboveLeftCandidate) {
            double pNextOB, qNextOB, pNextDB, qNextDB;
            double cpNext, dpNext, cotNext;
            long next;
            long nextDestination;

            dpLeftCandidate = dotProduct(pLeftCandidateOB, qLeftCandidateOB, pLeftCandidateDB, qLeftCandidateDB);
            cotLeftCandidate = dpLeftCandidate / cpLeftCandidate;

            do {
                next = nextEdge(leftCandidateIdx, baseOriginIdx);
                nextDestination = otherPoint(next, baseOriginIdx);

                makeVector(nextDestination, baseOriginIdx, pNextOB, qNextOB);
                makeVector(nextDestination, baseDestinationIdx, pNextDB, qNextDB);

                cpNext = crossProduct(pNextOB, qNextOB, pNextDB, qNextDB);
                aboveNext = cpNext > 0.0;

                if(!aboveNext) {
                    break;
                }

                dpNext = dotProduct(pNextOB, qNextOB, pNextDB, qNextDB);
                cotNext = dpNext / cpNext;
                
                if(cotNext > cotLeftCandidate) {
                    break;
                }
                
                deleteEdge(leftCandidateIdx);
                leftCandidateIdx = next;
                cotLeftCandidate = cotNext;
            } while(TRUE);

        }

        if(aboveRightCandidate) {
        
            double pPrevOB, qPrevOB, pPrevDB, qPrevDB;
            double cpPrev, dpPrev, cotPrev;
            long prev;
            long prevDestination;

            dpRightCandidate = dotProduct(pRightCandidateOB, qRightCandidateOB, pRightCandidateDB, qRightCandidateDB);
            cotRightCandidate = dpRightCandidate / cpRightCandidate;            

            do
            {
                
                prev = previousEdge(rightCandidateIdx, baseDestinationIdx);
                prevDestination = otherPoint(prev, baseDestinationIdx);
                
                makeVector(prevDestination, baseOriginIdx, pPrevOB, qPrevOB);
                makeVector(prevDestination, baseDestinationIdx, pPrevDB, qPrevDB);

                cpPrev = crossProduct(pPrevOB, qPrevOB, pPrevDB, qPrevDB);
                abovePrev = cpPrev > 0.0;

                if(!abovePrev) {
                    break;
                }
    
                dpPrev = dotProduct(pPrevOB, qPrevOB, pPrevDB, qPrevDB);
                cotPrev = dpPrev / cpPrev;

                if(cotPrev > cotRightCandidate) {
                    break;
                }

                deleteEdge(rightCandidateIdx);
                rightCandidateIdx = prev;
                cotRightCandidate = cotPrev;
                
            } while(TRUE);

        }            

        leftCandidateDestinationIdx = otherPoint(leftCandidateIdx, baseOriginIdx);
        rightCandidateDestinationIdx = otherPoint(rightCandidateIdx, baseDestinationIdx);
        if(!aboveLeftCandidate || (aboveLeftCandidate && aboveRightCandidate && cotRightCandidate < cotLeftCandidate) ) {
            base = makeTriangle(base, baseOriginIdx, rightCandidateIdx, rightCandidateDestinationIdx, 1); // side = right
            baseDestinationIdx = rightCandidateDestinationIdx;
        } else {
            base = makeTriangle(leftCandidateIdx, leftCandidateDestinationIdx, base, baseDestinationIdx, 1); // side = right
            baseOriginIdx = leftCandidateDestinationIdx;
        }
        
    } while(TRUE);    
    
}

void deleteEdge(long e)
{
    long p, q;

    p = edgeSet.edges[e].origin;
    q = edgeSet.edges[e].destination;

    if(pointSet.points[p].entryPoint == e) {
        pointSet.points[p].entryPoint = edgeSet.edges[e].originNext;
    }
    if(pointSet.points[q].entryPoint == e) {
        pointSet.points[q].entryPoint = edgeSet.edges[e].destinationNext;
    }

    if(edgeSet.edges[edgeSet.edges[e].originNext].origin == p) {
        edgeSet.edges[edgeSet.edges[e].originNext].originPrev = edgeSet.edges[e].originPrev;
    } else {
        edgeSet.edges[edgeSet.edges[e].originNext].destinationPrev = edgeSet.edges[e].originPrev;
    }

    if(edgeSet.edges[edgeSet.edges[e].originPrev].origin == p) {
        edgeSet.edges[edgeSet.edges[e].originPrev].originNext = edgeSet.edges[e].originNext;
    } else {
        edgeSet.edges[edgeSet.edges[e].originPrev].destinationNext = edgeSet.edges[e].originNext;
    }

    if(edgeSet.edges[edgeSet.edges[e].destinationNext].origin == q) {
        edgeSet.edges[edgeSet.edges[e].destinationNext].originPrev = edgeSet.edges[e].destinationPrev;
    } else {
        edgeSet.edges[edgeSet.edges[e].destinationNext].destinationPrev = edgeSet.edges[e].destinationPrev;
    }

    if(edgeSet.edges[edgeSet.edges[e].destinationPrev].origin == q) {
        edgeSet.edges[edgeSet.edges[e].destinationPrev].originNext = edgeSet.edges[e].destinationNext;
    } else {
        edgeSet.edges[edgeSet.edges[e].destinationPrev].destinationNext = edgeSet.edges[e].destinationNext;
    }

    

    edgeSet.edges[e].origin = -1;
    edgeSet.edges[e].destination = -1;
    edgeSet.edges[e].originNext = -1;
    edgeSet.edges[e].destinationNext = -1;
    edgeSet.edges[e].originPrev = -1;
    edgeSet.edges[e].destinationPrev = -1;

}

void printEdges(long myStart, long myEnd)
{
    long eStart, e;
    Point *u, *v;
    long i;
    long temp;

    for(i=myStart;i<myEnd;i++) {
        u = &pointSet.points[i];
        eStart = e = pointSet.points[i].entryPoint;
        do {
            temp = otherPoint(e,i);

            v = &pointSet.points[temp];
            if(u < v) {
                printf("%ld %ld\n", u-pointSet.points, v-pointSet.points);
            }
            e = nextEdge(e, i);
        } while(e != eStart);
    }       
}

void writeEdges(const char* filename, long myStart, long myEnd)
{
    FILE* f = fopen(filename, "w");
   
    if(f == NULL) {
        // logger(ERROR, "Failed to write to file");
        exit(1);
    }

    long eStart, e;
    Point *u, *v;
    long i;
    long temp;

    for(i=myStart;i<myEnd;i++) {
        u = &pointSet.points[i];
        eStart = e = pointSet.points[i].entryPoint;
        do {
            temp = otherPoint(e,i);

            v = &pointSet.points[temp];
            if(u < v) {
                fprintf(f, "%ld %ld\n", u-pointSet.points, v-pointSet.points);
            }
            e = nextEdge(e, i);
        } while(e != eStart);
    }
    fclose(f);
}


void printEdge(long e)
{
    printf("[origin:%ld; destination:%ld; originNext:%ld; originPrev:%ld; destinationNext:%ld; destinationPrev:%ld]\n",
            edgeSet.edges[e].origin, edgeSet.edges[e].destination, edgeSet.edges[e].originNext, 
            edgeSet.edges[e].originPrev, edgeSet.edges[e].destinationNext, edgeSet.edges[e].destinationPrev);
}

void printPoint(long p)
{
    printf("[x:%lf, y=%lf, entryPoint:%ld]\n", pointSet.points[p].x, pointSet.points[p].y, pointSet.points[p].entryPoint);
}
