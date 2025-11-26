#pragma once

#include "DroneDynamics.h"
#include "ExplicitEuler.h"
#include "ImplicitEuler.h"
#include "ImplicitMidpointIRK.h"
#include "RungeKutta4.h"
#include "UrdfRig.h"

#include <Eigen/Dense>
#include <array>
#include <string>
#include <unordered_map>

class DroneSimulationApp {
public:
    DroneSimulationApp();

    bool initializeScene(const std::string& urdfPath);
    void step();

private:
    struct PIDGains {
        Eigen::Vector3d kp{Eigen::Vector3d::Constant(1.0)};
        Eigen::Vector3d ki{Eigen::Vector3d::Constant(0.0)};
        Eigen::Vector3d kd{Eigen::Vector3d::Constant(1.5)};
    };

    struct AttGains {
        Eigen::Vector3d kR{0.0, 0.0, 0.0};
        Eigen::Vector3d kOm{0.0, 0.0, 0.0};
    };

    void initRotorData();
    void updateController();
    double motorThrustMax() const;
    void logState() const;
    void logIntegratorSettings() const;
    double hoverThrust() const;
    void normalizeQuaternions();
    Eigen::Isometry3d baseTransform() const;
    std::unordered_map<std::string, double> jointAngles() const;

    DroneDynamics dynamics_;
    ExplicitEuler explicitEuler_;
    RungeKutta4 rk4_;
    ImplicitEuler implicitEuler_;
    ImplicitMidpointIRK irk_;
    IntegratorType integratorType_{IntegratorType::Rk4};
    Eigen::VectorXd state_;
    Eigen::Vector4d thrust_;
    UrdfRig rig_;
    double simTime_{0.0};
    double dt_{0.002};
    int substeps_{5};
    double nextLogTime_{0.0};
    const double logInterval_{0.1};
    Eigen::Vector3d positionRef_{Eigen::Vector3d(0.0, 0.0, 1.0)};
    Eigen::Vector3d integralError_{Eigen::Vector3d::Zero()};
    PIDGains gains_;
    AttGains attGains_;
    std::array<Eigen::Vector3d, 4> rotorPositions_{};
    std::array<Eigen::Vector3d, 4> rotorDirsB_{};
    Eigen::Vector3d gravity_{0.0, 0.0, -9.8066};
    double yawRef_{0.0};
};
