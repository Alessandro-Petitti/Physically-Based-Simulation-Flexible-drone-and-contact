#include "DroneDynamics.h"
#include "ExplicitEuler.h"
#include "ImplicitEuler.h"
#include "ImplicitMidpointIRK.h"
#include "RungeKutta4.h"
#include "HullLoader.h"
#include "SceneUtils.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>

namespace {

double computeBaseZOffset(const ConvexHullShapes& hulls) {
    double zmin = std::numeric_limits<double>::infinity();
    for (const auto& v : hulls.baseHull_B) {
        zmin = std::min(zmin, v.z());
    }
    if (!std::isfinite(zmin)) return 0.0;
    return -zmin; // how much to raise to bring lowest point to z=0
}

void normalizeQuaternions(Eigen::VectorXd& state) {
    auto normalizeSegment = [&state](int offset) {
        Eigen::Vector4d q = state.segment<4>(offset);
        double n = q.norm();
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

} // namespace

int main(int argc, char** argv) {
    try {
        double dropHeight = 1.0; // desired clearance between lowest hull point and plane
        double maxTime = 5.0;
        if (argc == 2) {
            dropHeight = std::stod(argv[1]);
        } else if (argc == 3) {
            dropHeight = std::stod(argv[1]);
            maxTime = std::stod(argv[2]);
        } else if (argc != 1) {
            std::cout << "Usage: " << argv[0] << " [drop_height [max_time]]\n";
            return 1;
        }

        const std::string paramsPath = scene::resolveResource("model/drone_parameters.yaml").string();
        DroneDynamics dynamics(paramsPath);
        double hullScale = 0.001;
        if (const char* env = std::getenv("MORPHY_HULL_SCALE")) {
            try { hullScale = std::stod(env); } catch (...) {}
        }
        const auto hulls = loadConvexHullShapes(scene::resolveResource("graphics/hulls").string(), hullScale);
        const double baseOffset = computeBaseZOffset(hulls);

        ExplicitEuler explicitEuler;
        RungeKutta4 rk4;
        ImplicitEuler implicitEuler;
        ImplicitMidpointIRK irk;
        const auto& settings = dynamics.params().integratorSettings;
        const double dt = (settings.dt > 0.0) ? settings.dt : 0.002;
        const int substeps = std::max(1, settings.substeps);
        const int maxIt = std::max(1, settings.implicitMaxIterations);
        const double tol = (settings.implicitTolerance > 0.0) ? settings.implicitTolerance : 1e-6;
        const double fdEps = (settings.implicitFdEps > 0.0) ? settings.implicitFdEps : 1e-6;
        implicitEuler.setConfig(maxIt, tol, fdEps);
        irk.setConfig(maxIt, tol, fdEps);

        Eigen::VectorXd state = Eigen::VectorXd::Zero(DroneDynamics::kStateSize);
        state(2) = dropHeight + baseOffset; // position z
        state(3) = 1.0; // base quaternion w
        for (int i = 0; i < 4; ++i) {
            state(13 + 4 * i) = 1.0; // arm quaternions w
        }

        Eigen::Vector4d thrust = Eigen::Vector4d::Zero();
        auto dyn = [&](const Eigen::VectorXd& x) {
            return dynamics.derivative(x, thrust);
        };

        std::cout << "Contact drop test | drop height: " << dropHeight
                  << " m | dt: " << dt << " | substeps: " << substeps << "\n";

        double t = 0.0;
        double nextLog = 0.0;
        const double logInterval = 0.05;
        bool contacted = false;
        double contactTime = 0.0;

        while (t < maxTime) {
            for (int s = 0; s < substeps; ++s) {
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
                state = nextState;
                t += dt;

                const double zLowest = state(2) + (-baseOffset); // lowest hull point height
                if (!contacted && zLowest <= 0.0) {
                    contacted = true;
                    contactTime = t;
                    std::cout << "Contact detected at t = " << contactTime
                              << " s | z_lowest = " << zLowest << " m\n";
                }

                if (t >= nextLog) {
                    const Eigen::Vector3d p = state.segment<3>(0);
                    const Eigen::Vector3d v = state.segment<3>(7);
                    std::cout << std::fixed;
                    std::cout << "t = " << t << " s | z = " << p.z()
                              << " m | vz = " << v.z() << " m/s\n";
                    nextLog += logInterval;
                }
            }
        }

        const Eigen::Vector3d pFinal = state.segment<3>(0);
        const Eigen::Vector3d vFinal = state.segment<3>(7);
        std::cout << "\nEnd of simulation | t = " << t << " s\n";
        std::cout << "Final z = " << pFinal.z() << " m | vz = " << vFinal.z() << " m/s\n";
        if (!contacted) {
            std::cout << "No contact detected (likely still above ground).\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "contact_drop failed: " << e.what() << "\n";
        return 1;
    }
}
