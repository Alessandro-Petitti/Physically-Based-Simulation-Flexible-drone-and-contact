#pragma once

#include "SimpleYaml.h"

#include <Eigen/Dense>
#include <array>
#include <optional>
#include <string>

struct DroneParameters {
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
    std::array<Eigen::Matrix4d, 4> T_BH{};
    std::array<Eigen::Matrix4d, 4> T_HP{};
    std::array<Eigen::Matrix4d, 4> T_BP{};
};

DroneParameters loadDroneParameters(const std::string& path);
