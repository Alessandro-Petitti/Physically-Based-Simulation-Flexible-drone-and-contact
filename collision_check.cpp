#include "DroneDynamics.h"
#include "ExplicitEuler.h"
#include "ImplicitEuler.h"
#include "ImplicitMidpointIRK.h"
#include "RungeKutta4.h"
#include "SceneUtils.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

namespace {
const char* integratorLabel(IntegratorType type) {
    switch (type) {
        case IntegratorType::ExplicitEuler: return "explicit_euler";
        case IntegratorType::ImplicitEuler: return "implicit_euler";
        case IntegratorType::ImplicitMidpoint: return "irk_implicit_midpoint";
        case IntegratorType::Rk4:
        default: return "rk4";
    }
}

void normalizeQuaternions(Eigen::VectorXd& state) {
    auto normalizeSegment = [&state](int offset) {
        Eigen::Vector4d q = state.segment<4>(offset);
        const double n = q.norm();
        if (n > 1e-9) {
            state.segment<4>(offset) = q / n;
        } else {
            state.segment<4>(offset) << 1.0, 0.0, 0.0, 0.0;
        }
    };
    normalizeSegment(3);
    for (int i = 0; i < 4; ++i) {
        normalizeSegment(13 + 4 * i);
    }
}

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [x y z [max_time]]\n"
              << "  Defaults: x=0, y=0, z=1, max_time=10\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        double startX = 0.0;
        double startY = 0.0;
        double startZ = 1.0;
        double maxTime = 10.0;

        if (argc == 2) {
            startZ = std::stod(argv[1]);
        } else if (argc == 4) {
            startX = std::stod(argv[1]);
            startY = std::stod(argv[2]);
            startZ = std::stod(argv[3]);
        } else if (argc == 5) {
            startX = std::stod(argv[1]);
            startY = std::stod(argv[2]);
            startZ = std::stod(argv[3]);
            maxTime = std::stod(argv[4]);
        } else if (argc != 1) {
            printUsage(argv[0]);
            return 1;
        }

        if (startZ <= 0.0) {
            std::cout << "Start position already at/below the plane (z = "
                      << startZ << "). Nothing to simulate.\n";
            return 0;
        }

        const std::filesystem::path paramsPath =
            scene::resolveResource("model/drone_parameters.yaml");

        DroneDynamics dynamics(paramsPath.string());
        const auto& settings = dynamics.params().integratorSettings;
        const double dt = (settings.dt > 0.0) ? settings.dt : 0.002;
        const int substeps = std::max(1, settings.substeps);
        const int maxIterations = std::max(1, settings.implicitMaxIterations);
        const double tol = (settings.implicitTolerance > 0.0) ? settings.implicitTolerance : 1e-6;
        const double fdEps = (settings.implicitFdEps > 0.0) ? settings.implicitFdEps : 1e-6;

        ExplicitEuler explicitEuler;
        RungeKutta4 rk4;
        ImplicitEuler implicitEuler;
        ImplicitMidpointIRK irk;
        implicitEuler.setConfig(maxIterations, tol, fdEps);
        irk.setConfig(maxIterations, tol, fdEps);

        Eigen::VectorXd state = Eigen::VectorXd::Zero(DroneDynamics::kStateSize);
        state(0) = startX;
        state(1) = startY;
        state(2) = startZ;
        state(3) = 1.0;
        for (int i = 0; i < 4; ++i) {
            state(13 + 4 * i) = 1.0;
        }

        Eigen::Vector4d thrust = Eigen::Vector4d::Zero();
        auto dyn = [&](const Eigen::VectorXd& x) {
            return dynamics.derivative(x, thrust);
        };

        std::cout << "Free-fall collision check | start: ("
                  << startX << ", " << startY << ", " << startZ << ") m"
                  << " | max_time: " << maxTime << " s\n";
        std::cout << "Integrator: " << integratorLabel(dynamics.params().integrator)
                  << " | dt: " << dt << " | substeps: " << substeps << std::endl;

        double t = 0.0;
        double nextLog = 0.0;
        const double logInterval = 0.05;
        bool collided = false;
        double collisionTime = 0.0;
        Eigen::Vector3d collisionPos = Eigen::Vector3d::Zero();
        Eigen::Vector3d collisionVel = Eigen::Vector3d::Zero();

        while (t < maxTime && !collided) {
            for (int s = 0; s < substeps && !collided; ++s) {
                const double stepStart = t;
                const Eigen::VectorXd prevState = state;

                Eigen::VectorXd nextState;
                switch (dynamics.params().integrator) {
                    case IntegratorType::ExplicitEuler:
                        nextState = explicitEuler.step(state, dt, dyn);
                        break;
                    case IntegratorType::ImplicitEuler:
                        nextState = implicitEuler.step(state, dt, dyn);
                        break;
                    case IntegratorType::ImplicitMidpoint:
                        nextState = irk.step(state, dt, dyn);
                        break;
                    case IntegratorType::Rk4:
                    default:
                        nextState = rk4.step(state, dt, dyn);
                        break;
                }

                normalizeQuaternions(nextState);

                t += dt;

                const double zBefore = prevState(2);
                const double zAfter = nextState(2);
                if (zAfter <= 0.0) {
                    const double denom = zBefore - zAfter;
                    double alpha = (std::abs(denom) > 1e-9) ? zBefore / denom : 0.0;
                    alpha = std::clamp(alpha, 0.0, 1.0);
                    collisionTime = stepStart + alpha * dt;
                    t = collisionTime;
                    collisionPos = prevState.segment<3>(0)
                                 + alpha * (nextState.segment<3>(0) - prevState.segment<3>(0));
                    collisionVel = prevState.segment<3>(7)
                                 + alpha * (nextState.segment<3>(7) - prevState.segment<3>(7));
                    collided = true;
                    state = nextState;
                    break;
                }

                state = nextState;

                if (t >= nextLog) {
                    const Eigen::Vector3d p = state.segment<3>(0);
                    const Eigen::Vector3d v = state.segment<3>(7);
                    std::cout << std::fixed << std::setprecision(3)
                              << "t = " << t << " s | z = " << p.z()
                              << " m | vz = " << v.z() << " m/s" << std::endl;
                    nextLog += logInterval;
                }
            }
        }

        if (collided) {
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "\nCollision detected at t = " << collisionTime << " s\n";
            std::cout << "Contact point (approx): [" << collisionPos.transpose() << "] m\n";
            std::cout << "Velocity at contact: [" << collisionVel.transpose() << "] m/s\n";
            return 0;
        }

        const Eigen::Vector3d finalPos = state.segment<3>(0);
        std::cout << "\nNo collision within " << maxTime
                  << " s. Final z = " << finalPos.z() << " m\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Collision check failed: " << e.what() << std::endl;
        return 1;
    }
}
