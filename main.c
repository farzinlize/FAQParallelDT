#include<stdio.h>
#include"datasetlib.h"
#include"delaunay.h"

PointSet pointSet;
EdgeSet edgeSet;

void readOrFail(int argc, char **argv)
{

    if(argc < 4) {
        printf("Usage: main <data_file_as_csv> <num_of_points> <name_of_output_file>\n");
        exit(1);
    }

    char *inputFileName = argv[1];
    long numPoints = atol(argv[2]);
    long temp, i = 0,j=0;
    long numEdges = 0;

    FILE *inputFile = fopen(inputFileName, "r");
    // logger(INFO, "Opened input file. About to allocate memory");

    pointSet.points = malloc(sizeof(Point) * numPoints);
    // logger(INFO, "Memory request for Points complete");
    if(pointSet.points == NULL)
    {
        printf("Failed to allocate memory!\n");
        exit(3);
    }

    for(j=0;j<numPoints;j++) {
        pointSet.points[j].x = -1.00;
        pointSet.points[j].y = -1.00;
        pointSet.points[j].entryPoint = -1;
    }

    while(fscanf(inputFile, "%ld,%lf,%lf", &temp, &pointSet.points[i].x, &pointSet.points[i].y) == 3)
    {
        i++;
    }
    fclose(inputFile);

    printf("%ld points read into memory from file.\n", i);
    numEdges = 20;//logTwo(numPoints)*numPoints - 6;
    // logger(INFO, "About to allocate memory for edges");
    edgeSet.edges = malloc(sizeof(Edge) * numEdges);
    // logger(INFO, "Memory allocated for edges");
    if(edgeSet.edges == NULL)
    {
        // logger(ERROR, "Failed to get sufficient memory for Edges");
    }
    for(j=0;j<numEdges;j++) {
        edgeSet.edges[j].origin = -1;
        edgeSet.edges[j].destination = -1;
        edgeSet.edges[j].originNext = -1;
        edgeSet.edges[j].originPrev = -1;
        edgeSet.edges[j].destinationNext = -1;
        edgeSet.edges[j].destinationPrev = -1;
    }

    pointSet.size = numPoints;
    edgeSet.size = numEdges;

}

int main(int argc, char **argv)
{
    //pointSet pointSet = genrate_random_dataset();
    readOrFail(argc, argv);

    long leftEdgeIdx, rightEdgeIdx;

    delaunay(0, pointSet.size-1, &leftEdgeIdx, &rightEdgeIdx);

    writeEdges("res", 0, 5);

    printf("SUC\n");
    return 0;
}