#pragma once

#include "SimpleYaml.h"

#include <Eigen/Dense>
#include <array>
#include <optional>
#include <string>

enum class IntegratorType {
    ExplicitEuler,
    Rk4,
    ImplicitEuler,
    ImplicitMidpoint
};

struct DroneParameters {
    struct IntegratorSettings {
        double dt{0.002};
        int substeps{5};
        int implicitMaxIterations{8};
        double implicitTolerance{1e-6};
        double implicitFdEps{1e-6};
    };

    double massTotal{0.0};
    double massBase{0.0};
    double massArm{0.0};
    Eigen::Matrix3d inertiaBase{Eigen::Matrix3d::Identity()};
    Eigen::Matrix3d inertiaArm{Eigen::Matrix3d::Identity()};
    double kappaThrust{0.0};
    double kappaTorque{0.0};
    Eigen::Vector4d motorDirection{Eigen::Vector4d::Zero()};
    double rotorInertia{0.0};
    std::optional<double> propellerMaxThrust;
    double jointDamping{0.0};
    double jointStiffness{0.0};
    double contactStiffness{20.0};
    double contactDamping{0.5};
    double contactActivationDistance{0.0005};
    bool enableCCD{false};  // Continuous Collision Detection
    std::array<Eigen::Matrix4d, 4> T_BH{};
    std::array<Eigen::Matrix4d, 4> T_HP{};
    std::array<Eigen::Matrix4d, 4> T_BP{};
    IntegratorType integrator{IntegratorType::Rk4};
    IntegratorSettings integratorSettings{};
    Eigen::Vector3d x0_pos{Eigen::Vector3d::Zero()};
    Eigen::Vector4d x0_rotation{1.0, 0.0, 0.0, 0.0}; // wxyz
};

DroneParameters loadDroneParameters(const std::string& path);
