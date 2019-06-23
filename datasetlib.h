#ifndef _DATASETLIB_H_
#define _DATASETLIB_H_

typedef struct pointSet
{
    int * points;
    int size;
} pointSet;

typedef struct point
{
    int x, y;
} point;

pointSet genrate_random_dataset();
pointSet read_dataset_file(const char *);

#endif