#pragma once

#include <Eigen/Dense>

struct Plane {
    Eigen::Vector3d n; // unit normal pointing into the workspace
    double d;          // plane equation: n.dot(x) + d = 0
};

