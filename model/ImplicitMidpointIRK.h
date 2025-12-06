#pragma once

#include <Eigen/Dense>
#include <Eigen/LU>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

namespace implicit_midpoint_detail {
template <typename DynamicsFunc>
Eigen::MatrixXd numericalJacobian(const Eigen::VectorXd& x,
                                  const Eigen::VectorXd& fx,
                                  DynamicsFunc&& f,
                                  double eps) {
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();
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
    auto t1 = clock::now();
    double jac_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    //std::cout << "[TIMER] numericalJacobian took "<< jac_ms << " ms\n";
    return J;
}
} // namespace implicit_midpoint_detail

class ImplicitMidpointIRK {
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
        Eigen::VectorXd k = f(x); // initial guess from explicit Euler slope
        for (int iter = 0; iter < maxIterations_; ++iter) {
            const Eigen::VectorXd xStage = x + 0.5 * dt * k;
            const Eigen::VectorXd fStage = f(xStage);
            Eigen::VectorXd residual = k - fStage;
            if (!residual.allFinite()) {
                break;
            }
            if (residual.norm() < tolerance_) {
                break;
            }

            const Eigen::MatrixXd Jf = implicit_midpoint_detail::numericalJacobian(xStage, fStage, f, fdEps_);
            Eigen::MatrixXd G = Eigen::MatrixXd::Identity(n, n) - 0.5 * dt * Jf;
            const Eigen::VectorXd delta = G.fullPivLu().solve(-residual);
            Eigen::VectorXd k_new = k+delta;
            // line search
            double alpha = 1.0;
            double prevNorm = residual.norm();
            for (int ls = 0; ls < 10; ++ls) {
                Eigen::VectorXd xStageNew = x + 0.5 * dt * (k + alpha*delta);
                Eigen::VectorXd fStageNew = f(xStageNew);
                Eigen::VectorXd rNew = (k + alpha*delta) - fStageNew;
                if (rNew.norm() < prevNorm) {
                    k += alpha * delta;
                    residual = rNew;
                    break;
                }
                alpha *= 0.5;        // backtrack
            }
            //k += delta;
            if (!delta.allFinite() || delta.norm() < tolerance_) {
                break;
            }
        }
        return x + dt * k;
    }

private:
    int maxIterations_{8};
    double tolerance_{1e-6};
    double fdEps_{1e-6};
};
