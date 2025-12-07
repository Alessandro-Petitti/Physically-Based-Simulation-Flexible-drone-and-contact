#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>

#include <urdf_model/model.h>
#include <urdf_model/joint.h>
#include <urdf_model/link.h>
#include <urdf_model/pose.h>
#include <urdf_world/types.h>

namespace polyscope {
class SurfaceMesh;
}

class UrdfRig {
public:
    UrdfRig();

    bool initialize(const std::string& urdfPath);
    void update(const Eigen::Isometry3d& baseTransform,
                const std::unordered_map<std::string, double>& jointPositions);

    // Get the global transform (with viewAdjust applied) for a named link
    // Returns identity if not found
    Eigen::Matrix4f getLinkTransform(const std::string& linkName) const;

    // Get viewAdjust matrix
    const Eigen::Matrix4f& getViewAdjust() const { return viewAdjust_; }

private:
    struct LinkVisual {
        polyscope::SurfaceMesh* mesh{nullptr};
        Eigen::Matrix4f visualOffset{Eigen::Matrix4f::Identity()};
    };

    urdf::ModelInterfaceSharedPtr model_;
    std::map<std::string, LinkVisual> linkVisuals_;
    std::map<std::string, Eigen::Matrix4d> linkGlobalTransforms_;  // Global transforms per link
    Eigen::Matrix4f viewAdjust_{Eigen::Matrix4f::Identity()};
    std::filesystem::path projectRoot_;
    std::filesystem::path assetRoot_;

    void propagate(urdf::LinkConstSharedPtr link,
                   const Eigen::Matrix4d& parent,
                   const std::unordered_map<std::string, double>& jointPositions,
                   std::map<std::string, Eigen::Matrix4d>& global);

    Eigen::Matrix4d jointTransform(const urdf::JointConstSharedPtr& joint,
                                   double value) const;
};
