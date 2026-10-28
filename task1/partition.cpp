#include "partition.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

Range split_axis(int intervals, int parts, int coord) {
    if (intervals < 2 || parts < 1 || parts > intervals-1 || coord < 0 || coord >= parts)
        throw std::invalid_argument("invalid axis partition");
    int q = (intervals-1)/parts, r = (intervals-1)%parts;
    Range a;
    a.first = 1 + coord*q + std::min(coord,r);
    a.count = q + (coord < r ? 1 : 0);
    a.left = a.first-1;
    a.right = a.first+a.count-1 + (coord == parts-1 ? 1 : 0);
    return a;
}

Partition choose_partition(int M, int N, int processes) {
    if (M < 2 || N < 2 || processes < 1)
        throw std::invalid_argument("invalid grid or process count");
    if (static_cast<long long>(M-1)*(N-1) > std::numeric_limits<int>::max())
        throw std::invalid_argument("grid exceeds MPI integer counts");

    Partition best = {0,0};
    double best_score = std::numeric_limits<double>::infinity();
    for (int px = 1; px <= std::min(processes,M-1); ++px) {
        if (processes%px != 0) continue;
        int py = processes/px;
        if (py > N-1) continue;
        long long xmin = std::numeric_limits<long long>::max(), xmax = 0;
        long long ymin = std::numeric_limits<long long>::max(), ymax = 0;
        for (int x = 0; x < px; ++x) {
            Range a = split_axis(M,px,x);
            long long nodes = static_cast<long long>(a.right)-a.left+1;
            xmin = std::min(xmin,nodes);
            xmax = std::max(xmax,nodes);
        }
        for (int y = 0; y < py; ++y) {
            Range a = split_axis(N,py,y);
            long long nodes = static_cast<long long>(a.right)-a.left+1;
            ymin = std::min(ymin,nodes);
            ymax = std::max(ymax,nodes);
        }
        if (xmax > 2*ymin || ymax > 2*xmin) continue;
        double ratio = (static_cast<double>(M-1)/px)/(static_cast<double>(N-1)/py);
        double score = std::abs(std::log(ratio));
        if (score < best_score-1e-12 ||
            (std::abs(score-best_score) <= 1e-12 && px > best.px)) {
            best.px = px;
            best.py = py;
            best_score = score;
        }
    }
    if (best.px == 0)
        throw std::invalid_argument("no process grid satisfies domain aspect ratio [1/2,2]");
    return best;
}
