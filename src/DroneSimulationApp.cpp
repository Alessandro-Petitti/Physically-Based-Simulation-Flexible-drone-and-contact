#include "DroneSimulationApp.h"

#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <polyscope/view.h>
#include <polyscope/point_cloud.h>
#include <polyscope/surface_mesh.h>
#include "SceneUtils.h"
#include <nlohmann/json.hpp>

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
} // namespace

DroneSimulationApp::DroneSimulationApp()
    : dynamics_("model/drone_parameters.yaml"),
      state_(Eigen::VectorXd::Zero(DroneDynamics::kStateSize)) {
    state_.segment<3>(0) = dynamics_.params().x0_pos;
    state_.segment<4>(3) = dynamics_.params().x0_rotation;
    if (state_.segment<4>(3).norm() < 1e-9) {
        state_(3) = 1.0;
        state_(4) = state_(5) = state_(6) = 0.0;
    }
    for (int i = 0; i < 4; ++i) {
        state_(13 + 4 * i) = 1.0;
    }
    thrust_.setZero();
    integratorType_ = dynamics_.params().integrator;
    const auto& settings = dynamics_.params().integratorSettings;
    dt_ = (settings.dt > 0.0) ? settings.dt : 0.002;
    substeps_ = std::max(1, settings.substeps);
    const int maxIt = std::max(1, settings.implicitMaxIterations);
    const double tol = (settings.implicitTolerance > 0.0) ? settings.implicitTolerance : 1e-6;
    const double fdEps = (settings.implicitFdEps > 0.0) ? settings.implicitFdEps : 1e-6;
    implicitEuler_.setConfig(maxIt, tol, fdEps);
    irk_.setConfig(maxIt, tol, fdEps);
    logIntegratorSettings();
    std::cout << "Hover thrust per rotor: " << hoverThrust() << " N" << std::endl;
    initRotorData();

    planes_.push_back(Plane{Eigen::Vector3d(0.0, 0.0, 1.0), 0.0});
    // Load contact parameters from YAML
    const auto& params = dynamics_.params();
    contactParams_.contactStiffness = params.contactStiffness;
    contactParams_.contactDamping = params.contactDamping;
    contactParams_.activationDistance = params.contactActivationDistance;
    std::cout << "Contact params: k=" << contactParams_.contactStiffness 
              << " b=" << contactParams_.contactDamping 
              << " d=" << contactParams_.activationDistance << std::endl;
    if (const char* env = std::getenv("MORPHY_CONTACT_VIZ")) {
        enableContactViz_ = (std::string(env) != "0");
    }
    if (const char* env = std::getenv("MORPHY_START_CLEARANCE")) {
        try { startHeightClearance_ = std::stod(env); } catch (...) {}
    }
    if (const char* env = std::getenv("MORPHY_GROUND_Z")) {
        try { groundHeight_ = std::stod(env); } catch (...) {}
    }
    if (const char* env = std::getenv("MORPHY_DISABLE_PID")) {
        enableController_ = (std::string(env) == "0");
    }
    if (const char* env = std::getenv("MORPHY_FREE_FALL")) {
        if (std::string(env) == "1") enableController_ = false;
    }
    planes_.clear();
    planes_.push_back(Plane{Eigen::Vector3d(0.0, 0.0, 1.0), -groundHeight_});
}

bool DroneSimulationApp::initializeScene(const std::string& urdfPath) {
    if (!rig_.initialize(urdfPath)) {
        return false;
    }
    double hullScale = 0.001; // force millimeters -> meters
    hulls_ = loadConvexHullShapes(scene::resolveResource("graphics/hulls").string(), hullScale);
    hullMeshes_ = loadConvexHullMeshes(scene::resolveResource("graphics/hulls").string(), hullScale);
    // Ensure the drone is at least startHeightClearance_ above the ground
    const double zmin = baseHullZMin();
    state_(0) = 0.0;
    state_(1) = 0.0;
    if (std::isfinite(zmin)) {
        double minZ = groundHeight_ - zmin + startHeightClearance_;
        if (state_(2) < minZ) state_(2) = minZ;
    } else {
        if (state_(2) < groundHeight_ + startHeightClearance_) {
            state_(2) = groundHeight_ + startHeightClearance_;
        }
    }
    rig_.update(baseTransform(), jointAngles());
    initializeHullVisualization();
    initializeGroundPlaneVisualization();
    return true;
}

void DroneSimulationApp::step() {
    if (enableController_) {
        updateController();
    } else {
        thrust_.setZero(); // free-fall mode
    }

    auto dyn = [this](const Eigen::VectorXd& x) {
        return dynamics_.derivative(x, thrust_);
    };

    for (int i = 0; i < substeps_; ++i) {
        switch (integratorType_) {
            case IntegratorType::ExplicitEuler:
                state_ = explicitEuler_.step(state_, dt_, dyn);
                break;
            case IntegratorType::ImplicitEuler:
                state_ = implicitEuler_.step(state_, dt_, dyn);
                break;
            case IntegratorType::ImplicitMidpoint:
                state_ = irk_.step(state_, dt_, dyn);
                break;
            case IntegratorType::Rk4:
            default:
                state_ = rk4_.step(state_, dt_, dyn);
                break;
        }
        normalizeQuaternions();
        simTime_ += dt_;
    }

    rig_.update(baseTransform(), jointAngles());

    // Export transforms for the base and arms
    static int frameIndex = 0;
    namespace fs = std::filesystem;
    static bool exportInitialized = false;

    if (!exportInitialized) {
        const fs::path exportDir("export");
        std::error_code ec;
        fs::remove_all(exportDir, ec);
        if (ec) {
            std::cerr << "Warning: failed to clear export dir: " << ec.message() << '\n';
        }
        fs::create_directories(exportDir, ec);
        if (ec) {
            std::cerr << "Warning: failed to create export dir: " << ec.message() << '\n';
        }
        exportInitialized = true;
    }

    auto matrixToJson = [](const Eigen::Matrix4f& m) {
        nlohmann::json arr = nlohmann::json::array();
        for (int r = 0; r < 4; ++r) {
            nlohmann::json row = nlohmann::json::array();
            for (int c = 0; c < 4; ++c) {
                row.push_back(m(r, c));
            }
            arr.push_back(row);
        }
        return arr;
    };

    static const std::array<std::string, 4> armLinkNames = {
        "arm_motor_0", "arm_motor_1", "arm_motor_2", "arm_motor_3"
    };

    nlohmann::json armsLinkWorld = nlohmann::json::object();
    nlohmann::json armsVisualWorld = nlohmann::json::object();
    nlohmann::json armsViewFrame = nlohmann::json::object();
    nlohmann::json armsRelBase = nlohmann::json::object();
    nlohmann::json armsRelParent = nlohmann::json::object();
    for (int i = 0; i < 4; ++i) {
        const std::string key = std::to_string(i);
        const std::string& linkName = armLinkNames[i];
        armsLinkWorld[key] = matrixToJson(rig_.getLinkTransformWorld(linkName));
        armsVisualWorld[key] = matrixToJson(rig_.getLinkVisualTransformWorld(linkName));
        armsViewFrame[key] = matrixToJson(rig_.getLinkTransform(linkName));
        armsRelBase[key] = matrixToJson(rig_.getLinkTransformRelativeToBase(linkName));
        armsRelParent[key] = matrixToJson(rig_.getLinkTransformInParentFrame(linkName));
    }

    nlohmann::json payload;
    payload["view_adjust"] = matrixToJson(rig_.getViewAdjust());
    payload["base_link_world"] = matrixToJson(rig_.getLinkTransformWorld("base_link"));
    payload["base_visual_world"] = matrixToJson(rig_.getLinkVisualTransformWorld("base_link"));
    payload["base_view_frame"] = matrixToJson(rig_.getLinkTransform("base_link"));
    // Default keys for convenience/back-compat
    payload["base"] = payload["base_link_world"];
    payload["arms"] = armsLinkWorld;
    // Explicit buckets
    payload["arms_link_world"] = armsLinkWorld;
    payload["arms_visual_world"] = armsVisualWorld;
    payload["arms_view_frame"] = armsViewFrame;
    payload["arms_relative_to_base"] = armsRelBase;
    payload["arms_in_parent_frame"] = armsRelParent;

    std::ostringstream name;
    name << "export/frame_" << std::setfill('0') << std::setw(4) << frameIndex++ << ".json";

    std::ofstream file(name.str());
    if (file) {
        file << payload.dump(2);
    } else {
        std::cerr << "Failed to write " << name.str() << '\n';
    }

    if (simTime_ >= nextLogTime_) {
        logState();
        nextLogTime_ += logInterval_;
    }

    updateContactsVisualization();
    updateHullVisualization();
}

void DroneSimulationApp::logIntegratorSettings() const {
    const auto& s = dynamics_.params().integratorSettings;
    std::cout << "Integrator: " << integratorLabel(integratorType_)
              << " | dt: " << dt_
              << " | substeps: " << substeps_;
    if (integratorType_ == IntegratorType::ImplicitEuler ||
        integratorType_ == IntegratorType::ImplicitMidpoint) {
        std::cout << " | implicit iters: " << s.implicitMaxIterations
                  << " | tol: " << s.implicitTolerance
                  << " | fd_eps: " << s.implicitFdEps;
    }
    std::cout << std::endl;
}

void DroneSimulationApp::initRotorData() {
    const auto& params = dynamics_.params();
    Eigen::Vector3d ez(0.0, 0.0, 1.0);
    for (int i = 0; i < 4; ++i) {
        rotorPositions_[i] = params.T_BP[i].block<3,1>(0,3);
        Eigen::Matrix3d R_BP = params.T_BP[i].block<3,3>(0,0);
        rotorDirsB_[i] = R_BP * ez;
    }
}

void DroneSimulationApp::updateController() {
    Eigen::Vector3d p = state_.segment<3>(0);
    Eigen::Vector3d v = state_.segment<3>(7);

    Eigen::Vector3d errorPos = positionRef_ - p;
    integralError_ += errorPos * (dt_ * substeps_);
    Eigen::Vector3d errorVel = -v;

    Eigen::Vector3d accelCmd = gains_.kp.cwiseProduct(errorPos)
                               + gains_.ki.cwiseProduct(integralError_)
                               + gains_.kd.cwiseProduct(errorVel)
                               - gravity_;

    Eigen::Vector3d F_des = dynamics_.params().massTotal * accelCmd;

    Eigen::Vector3d b3d = accelCmd;
    if (b3d.norm() < 1e-6) b3d = Eigen::Vector3d(0,0,1);
    b3d.normalize();
    Eigen::Vector3d a_yaw(std::cos(yawRef_), std::sin(yawRef_), 0.0);
    Eigen::Vector3d b2d = b3d.cross(a_yaw);
    if (b2d.norm() < 1e-6) b2d = Eigen::Vector3d(0,1,0);
    b2d.normalize();
    Eigen::Vector3d b1d = b2d.cross(b3d);
    Eigen::Matrix3d R_d;
    R_d.col(0) = b1d;
    R_d.col(1) = b2d;
    R_d.col(2) = b3d;

    Eigen::Quaterniond q(state_(3), state_(4), state_(5), state_(6));
    if (q.norm() > 1e-9) q.normalize();
    Eigen::Matrix3d R = q.toRotationMatrix();
    Eigen::Matrix3d R_err = R_d.transpose() * R;
    Eigen::Vector3d e_R = 0.5 * Eigen::Vector3d(R_err(2,1) - R_err(1,2),
                                                R_err(0,2) - R_err(2,0),
                                                R_err(1,0) - R_err(0,1));
    Eigen::Vector3d w = state_.segment<3>(10);
    Eigen::Vector3d e_w = w;
    Eigen::Vector3d tau_cmd = - attGains_.kR.cwiseProduct(e_R) - attGains_.kOm.cwiseProduct(e_w);

    Eigen::Vector3d F_body = R.transpose() * F_des;
    Eigen::Matrix<double,6,4> H;
    double torqueRatio = dynamics_.params().kappaTorque / (dynamics_.params().kappaThrust + 1e-9);
    for (int i = 0; i < 4; ++i) {
        Eigen::Vector3d e_i = rotorDirsB_[i];
        Eigen::Vector3d r_i = rotorPositions_[i];
        H.block<3,1>(0,i) = e_i;
        H.block<3,1>(3,i) = r_i.cross(e_i) + dynamics_.params().motorDirection[i] * torqueRatio * e_i;
    }
    Eigen::Matrix<double,6,1> wrench;
    wrench << F_body, tau_cmd;

    Eigen::Vector4d u;
    Eigen::JacobiSVD<Eigen::Matrix<double,6,4>> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    u = svd.solve(wrench);
    for (int i = 0; i < 4; ++i) {
        u[i] = std::clamp(u[i], 0.0, motorThrustMax());
    }
    thrust_ = u;
}

double DroneSimulationApp::motorThrustMax() const {
    const auto& params = dynamics_.params();
    return params.propellerMaxThrust.value_or(10.0);
}

void DroneSimulationApp::logState() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "\n=== t = " << simTime_ << " s ===\n";
    Eigen::Vector3d p = state_.segment<3>(0);
    Eigen::Vector3d v = state_.segment<3>(7);
    Eigen::Vector3d w = state_.segment<3>(10);
    Eigen::Vector4d q = state_.segment<4>(3);
    oss << "p [m]: " << p.transpose() << " | v [m/s]: " << v.transpose() << "\n";
    oss << "w [rad/s]: " << w.transpose() << " | q_base: " << q.transpose() << "\n";
    for (int i = 0; i < 4; ++i) {
        Eigen::Vector4d qi = state_.segment<4>(13 + 4 * i);
        Eigen::Vector3d wi = state_.segment<3>(29 + 3 * i);
        oss << "arm_" << i << " q: " << qi.transpose()
            << " | rel_w [rad/s]: " << wi.transpose() << "\n";
    }
    oss << "thrust [N]: " << thrust_.transpose() << "\n";
    oss << "contacts: " << lastContactCount_
        << " | |sumF|: " << lastContactForceSumNorm_
        << " N | max|F|: " << lastContactForceMax_ << " N\n";
    std::cout << oss.str() << std::endl;
}

double DroneSimulationApp::hoverThrust() const {
    return dynamics_.params().massTotal * 9.8066 / 4.0;
}

void DroneSimulationApp::normalizeQuaternions() {
    auto normalizeSegment = [this](int offset) {
        Eigen::Vector4d q = state_.segment<4>(offset);
        double n = q.norm();
        if (n > 1e-9) {
            state_.segment<4>(offset) = q / n;
        } else {
            state_.segment<4>(offset) << 1.0, 0.0, 0.0, 0.0;
        }
    };
    normalizeSegment(3);
    for (int i = 0; i < 4; ++i) {
        normalizeSegment(13 + 4 * i);
    }
}

Eigen::Isometry3d DroneSimulationApp::baseTransform() const {
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.translation() = state_.segment<3>(0);
    Eigen::Quaterniond q(state_(3), state_(4), state_(5), state_(6));
    if (q.norm() > 1e-9) q.normalize();
    T.linear() = q.toRotationMatrix();
    return T;
}

std::unordered_map<std::string, double> DroneSimulationApp::jointAngles() const {
    std::unordered_map<std::string, double> joints;
    const auto eulers = dynamics_.armEulerZYX(state_);
    for (int i = 0; i < 4; ++i) {
        joints["base_link_to_connecting_link_" + std::to_string(i) + "_z"] = eulers[i].x();
        joints["base_link_to_connecting_link_" + std::to_string(i) + "_y"] = eulers[i].y();
        joints["base_link_to_connecting_link_" + std::to_string(i) + "_x"] = eulers[i].z();
    }
    return joints;
}

void DroneSimulationApp::updateContactsVisualization() {
    if (hulls_.baseHull_B.empty()) {
        return;
    }
    const bool doDraw = enableContactViz_ && ((vizCounter_++ % vizSkip_) == 0);

    auto normalizeQuat = [](Eigen::Quaterniond& q) {
        if (q.norm() > 1e-9) q.normalize();
    };

    Eigen::Quaterniond q_base(state_(3), state_(4), state_(5), state_(6));
    normalizeQuat(q_base);
    std::array<Eigen::Quaterniond,4> armQuat;
    for (int i = 0; i < 4; ++i) {
        armQuat[i] = Eigen::Quaterniond(state_(13 + 4 * i),
                                        state_(14 + 4 * i),
                                        state_(15 + 4 * i),
                                        state_(16 + 4 * i));
        normalizeQuat(armQuat[i]);
    }

    Eigen::Matrix<double,3,4> P_omega_rel;
    for (int i = 0; i < 4; ++i) {
        P_omega_rel.col(i) = state_.segment<3>(29 + 3 * i);
    }

    const auto arms = dynamics_.computeArmFramesFromState(q_base, armQuat);
    const Eigen::Matrix3d R_WB = q_base.toRotationMatrix();
    const Eigen::Matrix3d R_BW = R_WB.transpose();
    const Eigen::Vector3d v_WB = state_.segment<3>(7);
    const Eigen::Vector3d w_B = state_.segment<3>(10);
    const Eigen::Vector3d W_omega_B = R_WB * w_B;

    Eigen::Matrix<double,3,4> W_omega_P;
    for (int i = 0; i < 4; ++i) {
        const Eigen::Matrix3d R_WP = arms[i].R_WP;
        const Eigen::Matrix3d R_PW = R_WP.transpose();
        const Eigen::Matrix3d R_PB = R_PW * R_WB;
        const Eigen::Vector3d P_omega_P_i = P_omega_rel.col(i) + R_PB * w_B;
        W_omega_P.col(i) = R_WP * P_omega_P_i;
    }

    lastContacts_ = computeContacts(
        arms.data(),
        state_.segment<3>(0),
        R_WB,
        v_WB,
        W_omega_B,
        W_omega_P,
        hulls_,
        planes_,
        contactParams_);

    lastContactCount_ = lastContacts_.size();
    Eigen::Vector3d sumF = Eigen::Vector3d::Zero();
    lastContactForceMax_ = 0.0;
    for (const auto& c : lastContacts_) {
        sumF += c.force_W;
        lastContactForceMax_ = std::max(lastContactForceMax_, c.force_W.norm());
    }
    lastContactForceSumNorm_ = sumF.norm();

    // Decimate if too many contacts to avoid bogging down the UI
    std::vector<ContactPoint> vizContacts;
    vizContacts.reserve(std::min<std::size_t>(lastContacts_.size(), maxVizContacts_));
    if (lastContacts_.size() > maxVizContacts_) {
        std::size_t step = std::max<std::size_t>(1, lastContacts_.size() / maxVizContacts_);
        for (std::size_t i = 0; i < lastContacts_.size(); i += step) {
            vizContacts.push_back(lastContacts_[i]);
        }
    } else {
        vizContacts = lastContacts_;
    }

    if (!doDraw) {
        return;
    }

    // viewAdjust: same as UrdfRig uses for URDF meshes visualization
    static const Eigen::Matrix3d viewAdjust = 
        Eigen::AngleAxisd(-M_PI / 2.0, Eigen::Vector3d::UnitX()).toRotationMatrix();

    std::vector<glm::vec3> pts;
    std::vector<glm::vec3> vecs;
    pts.reserve(vizContacts.size());
    vecs.reserve(vizContacts.size());
    for (const auto& c : vizContacts) {
        Eigen::Vector3d x_view = viewAdjust * c.x_W;
        Eigen::Vector3d f_view = viewAdjust * c.force_W;
        pts.emplace_back(static_cast<float>(x_view.x()),
                         static_cast<float>(x_view.y()),
                         static_cast<float>(x_view.z()));
        vecs.emplace_back(static_cast<float>(f_view.x()),
                          static_cast<float>(f_view.y()),
                          static_cast<float>(f_view.z()));
    }

    polyscope::PointCloud* pc = nullptr;
    if (polyscope::hasPointCloud("contact_points")) {
        polyscope::removePointCloud("contact_points");
    }
    pc = polyscope::registerPointCloud("contact_points", pts);
    pc->setPointRadius(0.001, true);
    pc->addVectorQuantity("contact_force", vecs, polyscope::VectorType::AMBIENT)
        ->setVectorLengthScale(0.03);
}

double DroneSimulationApp::baseHullZMin() const {
    double zmin = std::numeric_limits<double>::infinity();
    for (const auto& v : hulls_.baseHull_B) {
        zmin = std::min(zmin, v.z());
    }
    return zmin;
}

void DroneSimulationApp::initializeHullVisualization() {
    if (!enableHullViz_) return;

    // Helper to convert vertices to glm::vec3
    auto toGlm = [](const std::vector<Eigen::Vector3d>& verts) {
        std::vector<glm::vec3> result;
        result.reserve(verts.size());
        for (const auto& v : verts) {
            result.emplace_back(static_cast<float>(v.x()),
                                static_cast<float>(v.y()),
                                static_cast<float>(v.z()));
        }
        return result;
    };

    // Register base hull as surface mesh
    auto baseVerts = toGlm(hullMeshes_.baseHull_B.vertices);
    auto* baseMesh = polyscope::registerSurfaceMesh("hull_base", baseVerts, hullMeshes_.baseHull_B.faces);
    baseMesh->setSurfaceColor({0.2f, 0.8f, 0.2f});
    baseMesh->setTransparency(0.5f);

    // Register arm hulls as surface meshes
    for (int i = 0; i < 4; ++i) {
        auto armVerts = toGlm(hullMeshes_.armHull_P[i].vertices);
        std::string name = "hull_arm_" + std::to_string(i);
        auto* armMesh = polyscope::registerSurfaceMesh(name, armVerts, hullMeshes_.armHull_P[i].faces);
        armMesh->setSurfaceColor({0.8f, 0.4f, 0.1f});
        armMesh->setTransparency(0.5f);
    }
}

void DroneSimulationApp::updateHullVisualization() {
    if (!enableHullViz_) return;

    // Use the transforms from UrdfRig (which handles the URDF joint chain correctly)
    // This ensures hull visualization is perfectly aligned with URDF mesh visualization.

    // Alignment matrices for hull .obj files relative to URDF .stl files
    // Base: URDF applies rpy="0 0 1.57079" (+π/2 around Z) to core_battery_transformed.stl
    static const Eigen::Matrix3f kBaseAlign = 
        Eigen::AngleAxisf(static_cast<float>(M_PI / 2.0), Eigen::Vector3f::UnitZ()).toRotationMatrix();
    // Arms: arm_transformed_hull.obj matches arm_transformed.stl
    static const Eigen::Matrix3f kArmAlign = Eigen::Matrix3f::Identity();

    // Get base transform from UrdfRig
    Eigen::Matrix4f T_base = rig_.getLinkTransform("base_link");
    Eigen::Matrix3f R_base = T_base.block<3,3>(0,0);
    Eigen::Vector3f t_base = T_base.block<3,1>(0,3);

    // Update base hull vertices
    std::vector<glm::vec3> baseVerts;
    baseVerts.reserve(hullMeshes_.baseHull_B.vertices.size());
    for (const auto& p_B : hullMeshes_.baseHull_B.vertices) {
        Eigen::Vector3f p_Bf = p_B.cast<float>();
        Eigen::Vector3f x_view = t_base + R_base * (kBaseAlign * p_Bf);
        baseVerts.emplace_back(x_view.x(), x_view.y(), x_view.z());
    }
    if (polyscope::hasSurfaceMesh("hull_base")) {
        polyscope::getSurfaceMesh("hull_base")->updateVertexPositions(baseVerts);
    }

    // Update arm hull vertices using the exact same transforms as URDF visualization
    static const std::array<std::string, 4> armLinkNames = {
        "arm_motor_0", "arm_motor_1", "arm_motor_2", "arm_motor_3"
    };

    for (int i = 0; i < 4; ++i) {
        Eigen::Matrix4f T_arm = rig_.getLinkTransform(armLinkNames[i]);
        Eigen::Matrix3f R_arm = T_arm.block<3,3>(0,0);
        Eigen::Vector3f t_arm = T_arm.block<3,1>(0,3);

        std::vector<glm::vec3> armVerts;
        armVerts.reserve(hullMeshes_.armHull_P[i].vertices.size());
        for (const auto& p_P : hullMeshes_.armHull_P[i].vertices) {
            Eigen::Vector3f p_Pf = p_P.cast<float>();
            Eigen::Vector3f x_view = t_arm + R_arm * (kArmAlign * p_Pf);
            armVerts.emplace_back(x_view.x(), x_view.y(), x_view.z());
        }
        std::string name = "hull_arm_" + std::to_string(i);
        if (polyscope::hasSurfaceMesh(name)) {
            polyscope::getSurfaceMesh(name)->updateVertexPositions(armVerts);
        }
    }
}

void DroneSimulationApp::initializeGroundPlaneVisualization() {
    // viewAdjust: same as UrdfRig uses for URDF meshes visualization
    static const Eigen::Matrix3d viewAdjust = 
        Eigen::AngleAxisd(-M_PI / 2.0, Eigen::Vector3d::UnitX()).toRotationMatrix();

    // Create a large quad at z = groundHeight_
    const double size = 2.0; // 2m x 2m plane
    std::vector<Eigen::Vector3d> corners = {
        {-size, -size, groundHeight_},
        { size, -size, groundHeight_},
        { size,  size, groundHeight_},
        {-size,  size, groundHeight_}
    };

    // Apply viewAdjust to match URDF visualization
    std::vector<glm::vec3> verts;
    verts.reserve(4);
    for (const auto& c : corners) {
        Eigen::Vector3d v = viewAdjust * c;
        verts.emplace_back(static_cast<float>(v.x()),
                           static_cast<float>(v.y()),
                           static_cast<float>(v.z()));
    }

    std::vector<std::vector<size_t>> faces = {{0, 1, 2, 3}};

    auto* mesh = polyscope::registerSurfaceMesh("ground_plane", verts, faces);
    mesh->setSurfaceColor({0.4f, 0.4f, 0.4f});
    mesh->setTransparency(0.7f);
}
