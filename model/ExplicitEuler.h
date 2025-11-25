#pragma once

#include <Eigen/Dense>

class ExplicitEuler {
public:
    template <typename DynamicsFunc>
    Eigen::VectorXd step(const Eigen::VectorXd& x,
                         double dt,
                         DynamicsFunc&& f) const {
        return x + dt * f(x);
    }
};
