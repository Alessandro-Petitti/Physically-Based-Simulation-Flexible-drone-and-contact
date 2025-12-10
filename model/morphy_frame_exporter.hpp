#pragma once

#include "ContactGeometry.h"
#include "DroneDynamics.h"

#include <Eigen/Geometry>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// Utility helpers to export Morphy link transforms to JSON files that Blender can consume.
// All quaternions are serialized in xyzw order.
namespace morphy::exporter {

struct LinkPose {
    Eigen::Vector3d p{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond q{Eigen::Quaterniond::Identity()};
};

struct ContactSample {
    std::string link;
    Eigen::Vector3d p{Eigen::Vector3d::Zero()};
    Eigen::Vector3d f{Eigen::Vector3d::Zero()};
};

struct FrameData {
    double t{0.0};
    int idx{0};
    std::unordered_map<std::string, LinkPose> poses;
    Eigen::Vector4d thrust{Eigen::Vector4d::Zero()};
    std::vector<ContactSample> contacts;
};

struct ExportConfig {
    double fps{60.0};
    std::string urdfPath;
    std::string meshDir;
    std::string framesSubdir{"frames"};
    std::array<Eigen::Vector3d, 4> armOffsets{}; // B frame offsets to each arm hinge
    Eigen::Vector3d armTipOffset{Eigen::Vector3d::Zero()}; // H frame offset to the motor tip
    std::vector<std::string> links{
        "base_link",
        "connecting_link_0", "connecting_link_1", "connecting_link_2", "connecting_link_3",
        "arm_motor_0", "arm_motor_1", "arm_motor_2", "arm_motor_3"
    };
    std::unordered_map<std::string, std::string> meshPaths;
};

inline Eigen::Quaterniond baseQuaternionFromState(const Eigen::VectorXd& state) {
    Eigen::Quaterniond q_base(state(3), state(4), state(5), state(6));
    if (q_base.norm() > 1e-9) {
        q_base.normalize();
    } else {
        q_base = Eigen::Quaterniond::Identity();
    }
    return q_base;
}

inline std::array<Eigen::Quaterniond, 4> armQuaternionsFromState(const Eigen::VectorXd& state) {
    std::array<Eigen::Quaterniond, 4> armQuat;
    for (int i = 0; i < 4; ++i) {
        const int offset = 13 + 4 * i;
        armQuat[i] = Eigen::Quaterniond(state(offset),
                                        state(offset + 1),
                                        state(offset + 2),
                                        state(offset + 3));
        if (armQuat[i].norm() > 1e-9) {
            armQuat[i].normalize();
        } else {
            armQuat[i] = Eigen::Quaterniond::Identity();
        }
    }
    return armQuat;
}

inline nlohmann::json vecToJson(const Eigen::Vector3d& v) {
    return nlohmann::json::array({v.x(), v.y(), v.z()});
}

inline nlohmann::json quatToJsonXYZW(const Eigen::Quaterniond& q_raw) {
    Eigen::Quaterniond q = q_raw;
    if (q.norm() > 1e-9) {
        q.normalize();
    } else {
        q = Eigen::Quaterniond::Identity();
    }
    return nlohmann::json::array({q.x(), q.y(), q.z(), q.w()});
}

inline FrameData buildFrame(const Eigen::VectorXd& state,
                            const DroneDynamics& dynamics,
                            double timeSeconds,
                            int frameIndex,
                            const Eigen::Vector4d& thrust,
                            const std::vector<ContactPoint>* contacts = nullptr) {
    if (state.size() != DroneDynamics::kStateSize) {
        throw std::runtime_error("Unexpected state vector size in buildFrame");
    }

    FrameData frame;
    frame.t = timeSeconds;
    frame.idx = frameIndex;
    frame.thrust = thrust;

    const Eigen::Vector3d p_WB = state.segment<3>(0);
    Eigen::Quaterniond q_base = baseQuaternionFromState(state);
    frame.poses["base_link"] = LinkPose{p_WB, q_base};

    const auto armQuat = armQuaternionsFromState(state);
    const auto arms = dynamics.computeArmFramesFromState(q_base, armQuat);

    for (int i = 0; i < 4; ++i) {
        const Eigen::Quaterniond q_WH(arms[i].R_WH);
        const Eigen::Quaterniond q_WP(arms[i].R_WP);

        frame.poses["connecting_link_" + std::to_string(i)] =
            LinkPose{p_WB + arms[i].W_r_BH, q_WH};
        frame.poses["arm_motor_" + std::to_string(i)] =
            LinkPose{p_WB + arms[i].W_r_BP, q_WP};
    }

    if (contacts) {
        frame.contacts.reserve(contacts->size());
        for (const auto& c : *contacts) {
            ContactSample sample;
            if (c.bodyId <= 0) {
                sample.link = "base_link";
            } else {
                const int armIdx = std::clamp(c.bodyId - 1, 0, 3);
                sample.link = "arm_motor_" + std::to_string(armIdx);
            }
            sample.p = c.x_W;
            sample.f = c.force_W;
            frame.contacts.push_back(sample);
        }
    }

    return frame;
}

inline bool writeFrameJson(const FrameData& frame,
                           const std::filesystem::path& framesDir) {
    std::error_code ec;
    std::filesystem::create_directories(framesDir, ec);

    nlohmann::json j;
    j["t"] = frame.t;
    j["idx"] = frame.idx;

    nlohmann::json poses = nlohmann::json::object();
    for (const auto& kv : frame.poses) {
        poses[kv.first] = {
            {"p", vecToJson(kv.second.p)},
            {"q", quatToJsonXYZW(kv.second.q)}
        };
    }
    j["T_W_L"] = poses;

    j["thrust"] = {frame.thrust[0], frame.thrust[1], frame.thrust[2], frame.thrust[3]};

    nlohmann::json contacts = nlohmann::json::array();
    for (const auto& c : frame.contacts) {
        contacts.push_back({
            {"link", c.link},
            {"p", vecToJson(c.p)},
            {"f", vecToJson(c.f)}
        });
    }
    j["contacts"] = contacts;

    std::ostringstream name;
    name << "frame_" << std::setfill('0') << std::setw(4) << frame.idx << ".json";
    const std::filesystem::path outPath = framesDir / name.str();

    std::ofstream out(outPath);
    if (!out) {
        return false;
    }
    out << j.dump(2);
    return true;
}

inline bool writeConfigJson(const ExportConfig& cfg,
                            const std::filesystem::path& outDir) {
    std::error_code ec;
    std::filesystem::create_directories(outDir, ec);

    nlohmann::json j;
    j["robot"] = "morphy";
    j["fps"] = cfg.fps;
    j["frames_dir"] = cfg.framesSubdir;
    if (!cfg.urdfPath.empty()) j["urdf"] = cfg.urdfPath;
    if (!cfg.meshDir.empty()) j["mesh_dir"] = cfg.meshDir;

    nlohmann::json offsets = nlohmann::json::array();
    for (const auto& off : cfg.armOffsets) {
        offsets.push_back(vecToJson(off));
    }
    j["arm_offsets_B"] = offsets;
    j["arm_tip_offset"] = vecToJson(cfg.armTipOffset);

    nlohmann::json links = nlohmann::json::array();
    for (const auto& l : cfg.links) {
        links.push_back(l);
    }
    j["links"] = links;

    if (!cfg.meshPaths.empty()) {
        nlohmann::json meshes = nlohmann::json::object();
        for (const auto& kv : cfg.meshPaths) {
            meshes[kv.first] = kv.second;
        }
        j["mesh_paths"] = meshes;
    }

    std::ofstream out(outDir / "config.json");
    if (!out) {
        return false;
    }
    out << j.dump(2);
    return true;
}

// Example usage (within a simulation loop):
//   static int frameIdx = 0;
//   static bool initialized = false;
//   static std::filesystem::path root{"animation_data"}, frames = root / "frames";
//   if (!initialized) {
//       morphy::exporter::ExportConfig cfg;
//       cfg.fps = 1.0 / (dt * substeps);
//       cfg.urdfPath = "/abs/path/to/graphics/urdf/morphy.urdf";
//       writeConfigJson(cfg, root);
//       initialized = true;
//   }
//   auto frame = morphy::exporter::buildFrame(state, dynamics, simTime, frameIdx++, thrust, &contacts);
//   morphy::exporter::writeFrameJson(frame, frames);

} // namespace morphy::exporter
