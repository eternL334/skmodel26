#ifndef PARTITION_H
#define PARTITION_H

struct Range {
    int first, count, left, right;
};

struct Partition {
    int px, py;
};

Range split_axis(int intervals, int parts, int coord);
Partition choose_partition(int M, int N, int processes);

#endif
