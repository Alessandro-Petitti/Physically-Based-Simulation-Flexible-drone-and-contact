#pragma once

#include <Eigen/Dense>
#include <Eigen/LU>

#include <algorithm>
#include <cmath>

namespace implicit_euler_detail {
template <typename DynamicsFunc>
Eigen::MatrixXd numericalJacobian(const Eigen::VectorXd& x,
                                  const Eigen::VectorXd& fx,
                                  DynamicsFunc&& f,
                                  double eps) {
    const int n = static_cast<int>(x.size());
    Eigen::MatrixXd J(n, n);
    Eigen::VectorXd xPert = x;
    for (int i = 0; i < n; ++i) {
        double h = eps * std::max(1.0, std::abs(xPert[i]));
        if (h == 0.0) h = eps;
        xPert[i] += h;
        const Eigen::VectorXd fPert = f(xPert);
        J.col(i) = (fPert - fx) / h;
        xPert[i] = x[i];
    }
    return J;
}
} // namespace implicit_euler_detail

class ImplicitEuler {
public:
    void setConfig(int maxIterations, double tolerance, double fdEps) {
        maxIterations_ = maxIterations;
        tolerance_ = tolerance;
        fdEps_ = fdEps;
    }

    template <typename DynamicsFunc>
    Eigen::VectorXd step(const Eigen::VectorXd& x,
                         double dt,
                         DynamicsFunc&& f) const {
        const int n = static_cast<int>(x.size());
        Eigen::VectorXd y = x;
        for (int iter = 0; iter < maxIterations_; ++iter) {
            const Eigen::VectorXd f_y = f(y);
            Eigen::VectorXd residual = y - x - dt * f_y;
            if (!residual.allFinite()) {
                break;
            }
            if (residual.norm() < tolerance_) {
                break;
            }

            const Eigen::MatrixXd Jf = implicit_euler_detail::numericalJacobian(y, f_y, f, fdEps_);
            Eigen::MatrixXd G = Eigen::MatrixXd::Identity(n, n) - dt * Jf;
            const Eigen::VectorXd delta = G.fullPivLu().solve(-residual);
            y += delta;
            if (!delta.allFinite() || delta.norm() < tolerance_) {
                break;
            }
        }
        return y;
    }

private:
    int maxIterations_{8};
    double tolerance_{1e-6};
    double fdEps_{1e-6};
};
