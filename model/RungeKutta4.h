#pragma once

#include <Eigen/Dense>
#include <functional>

class RungeKutta4 {
public:
    template <typename DynamicsFunc>
    Eigen::VectorXd step(const Eigen::VectorXd& x,
                         double dt,
                         DynamicsFunc&& f) const {
        const Eigen::VectorXd k1 = f(x);
        const Eigen::VectorXd k2 = f(x + 0.5 * dt * k1);
        const Eigen::VectorXd k3 = f(x + 0.5 * dt * k2);
        const Eigen::VectorXd k4 = f(x + dt * k3);
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }
};
