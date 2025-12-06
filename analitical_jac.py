#!/usr/bin/env python3
import sympy as sp
from sympy.utilities.codegen import codegen

############################################################
# 1. STATE VECTOR (match your C++ state layout)
############################################################

# In your C++:
#   pos (0:3), q_base (3:7), v (7:10), w (10:13),
#   4 arm quaternions (13..29), 4*3 relative omegas (29..41)
kStateSize = 41
x_syms = sp.symbols(f'x0:{kStateSize}')   # x0, x1, ..., x40
x = sp.Matrix(x_syms)

# Convenience slices (just names, more readable)
# NB: ranges are Python-style [start:end], end is excluded.
p = sp.Matrix(x_syms[0:3])       # position
qB = sp.Matrix(x_syms[3:7])      # base quaternion (w,x,y,z)
v = sp.Matrix(x_syms[7:10])      # linear velocity (world)
wB = sp.Matrix(x_syms[10:13])    # angular velocity (body)

# arm quaternions:
qA = [sp.Matrix(x_syms[13+4*i : 17+4*i]) for i in range(4)]
# relative joint angular velocities (in P-frame)
omega_rel = [sp.Matrix(x_syms[29+3*i : 32+3*i]) for i in range(4)]


############################################################
# 2. PARAMETERS (symbolic for now)
############################################################

# Example parameters – you can add as many as you need
mB, mP = sp.symbols('mB mP')  # base mass, arm mass
g = sp.symbols('g')           # gravity magnitude

# Inertia matrices for base and arms as full symmetric 3x3
Ibx, Iby, Ibz, Ibxy, Ibxz, Ibyz = sp.symbols(
    'Ibx Iby Ibz Ibxy Ibxz Ibyz'
)
Iax, Iay, Iaz, Iaxy, Iaxz, Iayz = sp.symbols(
    'Iax Iay Iaz Iaxy Iaxz Iayz'
)

I_B = sp.Matrix([
    [Ibx, Ibxy, Ibxz],
    [Ibxy, Iby, Ibyz],
    [Ibxz, Ibyz, Ibz]
])
I_A = sp.Matrix([
    [Iax, Iaxy, Iaxz],
    [Iaxy, Iay, Iayz],
    [Iaxz, Iayz, Iaz]
])

# Example: thrusts (you may treat them as extra inputs or parameters)
T0, T1, T2, T3 = sp.symbols('T0 T1 T2 T3')
thrust = sp.Matrix([T0, T1, T2, T3])

# World gravity vector
g_vec = sp.Matrix([0, 0, -g])

############################################################
# 3. HELPER FUNCTIONS: skew, quaternion math, etc.
############################################################

def skew(v3):
    """Return 3x3 skew-symmetric matrix [v]_x."""
    vx, vy, vz = v3
    return sp.Matrix([
        [0,   -vz,  vy],
        [vz,   0,  -vx],
        [-vy,  vx,  0]
    ])

def quat_to_rot(q):
    """Quaternion q = [w,x,y,z] -> 3x3 rotation matrix."""
    wq, xq, yq, zq = q
    R = sp.Matrix([[1 - 2*(yq**2 + zq**2),     2*(xq*yq - zq*wq),     2*(xq*zq + yq*wq)],
                   [2*(xq*yq + zq*wq),         1 - 2*(xq**2 + zq**2), 2*(yq*zq - xq*wq)],
                   [2*(xq*zq - yq*wq),         2*(yq*zq + xq*wq),     1 - 2*(xq**2 + yq**2)]])
    return R

def quat_omega_dot(q, omega):
    """
    Quaternion kinematics: q_dot = 0.5 * q ⊗ [0, omega]
    q = [w,x,y,z], omega = [wx,wy,wz]
    """
    wq, xq, yq, zq = q
    wx, wy, wz = omega

    # q ⊗ omega_q with omega_q = (0, wx, wy, wz)
    dq_w = -0.5*(xq*wx + yq*wy + zq*wz)
    dq_x =  0.5*(wq*wx + yq*wz - zq*wy)
    dq_y =  0.5*(wq*wy + zq*wx - xq*wz)
    dq_z =  0.5*(wq*wz + xq*wy - yq*wx)

    return sp.Matrix([dq_w, dq_x, dq_y, dq_z])

def log_SO3(R):
    """
    Lightweight SO(3) logarithm (small-angle friendly):
      phi = 0.5 * vee(R - R.T)
    This matches the small-angle branch used in the C++ code and keeps
    the symbolic expression compact.
    """
    vee = sp.Matrix([
        R[2,1] - R[1,2],
        R[0,2] - R[2,0],
        R[1,0] - R[0,1]
    ]) * sp.Rational(1, 2)
    return vee

############################################################
# 3b. Kinematics helpers mirroring the C++ code
############################################################

kEz = sp.Matrix([0, 0, 1])

def compute_arm_frames(qB, qA, params):
    """Replicate DroneDynamics::computeArmFrames, returning a list of dicts."""
    R_WB = quat_to_rot(qB)
    T_BH_list = params.get("T_BH", [sp.eye(4) for _ in range(4)])
    T_HP_list = params.get("T_HP", [sp.eye(4) for _ in range(4)])

    arms = []
    for i in range(4):
        T_BH = T_BH_list[i]
        T_HP = T_HP_list[i]

        R_BH0 = T_BH[:3, :3]
        B_r_BH = T_BH[:3, 3]
        R_H0H = quat_to_rot(qA[i])
        R_BH = R_BH0 * R_H0H
        R_HP = T_HP[:3, :3]
        H_r_HP = T_HP[:3, 3]

        R_WH = R_WB * R_BH
        R_WP = R_WH * R_HP
        W_r_BH = R_WB * B_r_BH
        W_r_HP = R_WH * H_r_HP
        W_r_BP = W_r_BH + W_r_HP

        arms.append({
            "R_BH0": R_BH0,
            "R_H0H": R_H0H,
            "R_BH": R_BH,
            "R_WH": R_WH,
            "R_WP": R_WP,
            "R_HP": R_HP,
            "W_r_BH": W_r_BH,
            "W_r_HP": W_r_HP,
            "W_r_BP": W_r_BP,
        })
    return arms

def compute_kinematics(x, params):
    """
    Pre-compute all frame quantities used in A,b assembly.
    Returns a dict to avoid recomputing inside build_f.
    """
    qB = sp.Matrix(x[3:7])
    wB = sp.Matrix(x[10:13])
    qA_list = [sp.Matrix(x[13+4*i : 17+4*i]) for i in range(4)]
    omega_rel_list = [sp.Matrix(x[29+3*i : 32+3*i]) for i in range(4)]

    R_WB = quat_to_rot(qB)
    R_BW = R_WB.T
    arms = compute_arm_frames(qB, qA_list, params)

    W_omega_B = R_WB * wB

    R_WP_list = []
    R_PB_list = []
    P_omega_rel_list = []
    P_omega_P_list = []
    W_omega_P_list = []
    WF_thrust_list = []

    thrust_vec = params.get("thrust", thrust)

    for i in range(4):
        arm = arms[i]
        R_WP = arm["R_WP"]
        R_WP_list.append(R_WP)
        R_PW = R_WP.T
        R_PB = R_PW * R_WB
        R_PB_list.append(R_PB)

        omega_rel_i = omega_rel_list[i]
        P_omega_rel_list.append(omega_rel_i)

        P_omega_P_i = omega_rel_i + R_PB * wB
        P_omega_P_list.append(P_omega_P_i)
        W_omega_P_list.append(R_WP * P_omega_P_i)

        WF_thrust_list.append(R_WP * kEz * thrust_vec[i])

    return {
        "qB": qB,
        "wB": wB,
        "R_WB": R_WB,
        "R_BW": R_BW,
        "arms": arms,
        "W_omega_B": W_omega_B,
        "R_WP": R_WP_list,
        "R_PB": R_PB_list,
        "P_omega_rel": P_omega_rel_list,
        "P_omega_P": P_omega_P_list,
        "W_omega_P": W_omega_P_list,
        "WF_thrust": WF_thrust_list,
    }

def compute_arm_torques(kin, params):
    """
    Compute symbolic torques P_tau_SD, P_tau_drag, P_tau_gyro for each arm.
    Uses a small-angle log map for stiffness term.
    """
    joint_damp = params.get("joint_damping", sp.Symbol("b_joint"))
    joint_stiff = params.get("joint_stiffness", sp.Symbol("k_joint"))
    kappa_thrust = params.get("kappa_thrust", sp.Symbol("kappa_thrust"))
    kappa_torque = params.get("kappa_torque", sp.Symbol("kappa_torque"))
    rotor_inertia = params.get("rotor_inertia", sp.Symbol("J_r"))
    motor_dir = params.get("motor_dir", [sp.Symbol(f"d{i}") for i in range(4)])

    P_tau_SD = sp.zeros(3, 4)
    P_tau_drag = sp.zeros(3, 4)
    P_tau_gyro = sp.zeros(3, 4)

    thrust_vec = params.get("thrust", thrust)

    for i in range(4):
        arm = kin["arms"][i]
        R_H0H = arm["R_H0H"]
        R_PH = arm["R_HP"].T
        P_omega_rel_i = kin["P_omega_rel"][i]

        H0_phi = log_SO3(R_H0H)
        H_phi = R_H0H.T * H0_phi
        P_phi = R_PH * H_phi

        omegaSq = thrust_vec[i] / (kappa_thrust + sp.Float("1e-9"))
        spin = motor_dir[i] * sp.sqrt(omegaSq)
        h_i_P = rotor_inertia * spin * kEz

        P_tau_SD[:, i] = -(joint_damp * P_omega_rel_i) - (joint_stiff * P_phi)
        P_tau_drag[:, i] = -motor_dir[i] * kappa_torque * omegaSq * kEz
        P_tau_gyro[:, i] = P_omega_rel_i.cross(h_i_P)

    return P_tau_SD, P_tau_drag, P_tau_gyro

############################################################
# 4. DYNAMICS CORE: build A(x), b(x) symbolically
############################################################

def build_A_b(x, params):
    """
    Build 18x18 matrix A(x) and 18x1 vector b(x) symbolically.
    This should mirror your C++ assembly:
      - buildTranslationalBlock
      - buildBodyRotationalBlock
      - buildArmBlock (x4)
    The expressions below mirror the C++ code structurally,
    using a small-angle SO(3) log for stiffness terms to keep things symbolic.
    """

    kin = compute_kinematics(x, params)
    P_tau_SD, P_tau_drag, P_tau_gyro = compute_arm_torques(kin, params)

    mB_sym = params.get("mB", mB)
    mP_sym = params.get("mP", mP)
    I_B_sym = params.get("I_B", I_B)
    I_A_sym = params.get("I_A", I_A)

    A = sp.zeros(18, 18)
    b = sp.zeros(18, 1)
    row = 0

    # ----------------------------------------------------
    # Translational block (rows 0..2)
    # ----------------------------------------------------
    Mtot = mB_sym + 4 * mP_sym
    A_trans = sp.zeros(3, 18)
    b_trans = sp.zeros(3, 1)
    A_trans[:3, :3] = Mtot * sp.eye(3)
    b_trans += Mtot * g_vec

    for i in range(4):
        arm = kin["arms"][i]
        W_r_BH = arm["W_r_BH"]
        W_r_HP = arm["W_r_HP"]
        A_trans[:3, 3:6] += -mP_sym * skew(W_r_BH)
        col = 6 + 3 * i
        A_trans[:3, col:col+3] = -mP_sym * skew(W_r_HP)

        W_omega_B = kin["W_omega_B"]
        W_omega_P_i = kin["W_omega_P"][i]
        WF_thrust_i = kin["WF_thrust"][i]
        b_trans += WF_thrust_i
        b_trans += -mP_sym * (W_omega_B.cross(W_omega_B.cross(W_r_BH)))
        b_trans += -mP_sym * (W_omega_P_i.cross(W_omega_P_i.cross(W_r_HP)))

    A[row:row+3, :] = A_trans
    b[row:row+3, 0] = b_trans
    row += 3

    # ----------------------------------------------------
    # Body rotational block (rows 3..5)
    # ----------------------------------------------------
    R_WB = kin["R_WB"]
    R_BW = kin["R_BW"]
    W_omega_B = kin["W_omega_B"]
    B_omega_B = R_BW * W_omega_B

    A_body = sp.zeros(3, 18)
    b_body = -B_omega_B.cross(I_B_sym * B_omega_B)
    A_body[:, 3:6] = I_B_sym * R_BW

    for i in range(4):
        arm = kin["arms"][i]
        W_r_BH = arm["W_r_BH"]
        W_r_HP = arm["W_r_HP"]
        B_r_BH = R_BW * W_r_BH

        skew_term = skew(B_r_BH) * R_BW
        A_body[:, 0:3] += skew_term * (mP_sym * sp.eye(3))
        A_body[:, 3:6] += -skew_term * (skew(W_r_BH) * mP_sym)
        col = 6 + 3 * i
        A_body[:, col:col+3] = -skew_term * (skew(W_r_HP) * mP_sym)

        WF_thrust_i = kin["WF_thrust"][i]
        W_omega_P_i = kin["W_omega_P"][i]
        b_body += skew_term * (mP_sym * g_vec)
        b_body += skew_term * WF_thrust_i
        b_body += -skew_term * (mP_sym * (W_omega_B.cross(W_omega_B.cross(W_r_BH))))
        b_body += -skew_term * (mP_sym * (W_omega_P_i.cross(W_omega_P_i.cross(W_r_HP))))
        b_body += -(R_BW * (arm["R_WP"] * P_tau_SD[:, i]))

    A[row:row+3, :] = A_body
    b[row:row+3, 0] = b_body
    row += 3

    # ----------------------------------------------------
    # Arm blocks (rows 6..17)
    # ----------------------------------------------------
    for i in range(4):
        arm = kin["arms"][i]
        R_WP = arm["R_WP"]
        R_PW = R_WP.T
        R_PB = kin["R_PB"][i]

        W_r_BH = arm["W_r_BH"]
        W_r_HP = arm["W_r_HP"]
        P_r_PHi = R_PW * (-W_r_HP)

        W_omega_B = kin["W_omega_B"]
        W_omega_P_i = kin["W_omega_P"][i]

        A_arm = sp.zeros(3, 18)
        b_arm = sp.zeros(3, 1)

        col = 6 + 3 * i
        A_arm[:, col:col+3] = (I_A_sym * R_PW) + (skew(P_r_PHi) * R_PW) * (mP_sym * skew(W_r_HP))
        A_arm[:, 3:6] = (skew(P_r_PHi) * R_PW) * (mP_sym * skew(W_r_BH))
        A_arm[:, 0:3] = -(skew(P_r_PHi) * R_PW) * (mP_sym * sp.eye(3))

        P_omega_P_i = kin["P_omega_P"][i]
        P_tau_SD_i = P_tau_SD[:, i]
        P_tau_drag_i = P_tau_drag[:, i]
        P_tau_gyro_i = P_tau_gyro[:, i]
        WF_thrust_i = kin["WF_thrust"][i]

        b_arm += -P_omega_P_i.cross(I_A_sym * P_omega_P_i)
        b_arm += (skew(P_r_PHi) * R_PW) * (
            mP_sym * (W_omega_B.cross(W_omega_B.cross(W_r_BH)) +
                      W_omega_P_i.cross(W_omega_P_i.cross(W_r_HP)) - g_vec)
        )
        b_arm += -(skew(P_r_PHi) * R_PW) * WF_thrust_i
        b_arm += P_tau_SD_i + P_tau_drag_i + P_tau_gyro_i

        A[row:row+3, :] = A_arm
        b[row:row+3, 0] = b_arm
        row += 3

    return A, b, kin

############################################################
# 5. BUILD f(x): whole state derivative
############################################################

def build_f(x, params):
    """
    Build the full derivative f(x) symbolically, matching DroneDynamics::derivative.
    """
    # Unpack state again for readability
    p = sp.Matrix(x[0:3])
    qB = sp.Matrix(x[3:7])
    v = sp.Matrix(x[7:10])
    wB = sp.Matrix(x[10:13])
    qA = [sp.Matrix(x[13+4*i : 17+4*i]) for i in range(4)]
    omega_rel = [sp.Matrix(x[29+3*i : 32+3*i]) for i in range(4)]

    # Precompute kinematics and A,b
    A, b, kin = build_A_b(x, params)
    z = A.LUsolve(b)  # z is 18x1

    # Extract pieces of z: structure as in C++:
    W_a_B = z[0:3, 0]
    W_omega_dot_B = z[3:6, 0]
    W_omega_dot_P = [z[6+3*i : 9+3*i, 0] for i in range(4)]

    # You may need R_WB, R_BW etc, depending on frames
    R_WB = quat_to_rot(qB)
    R_BW = R_WB.T
    B_omega_dot_B = R_BW * W_omega_dot_B

    # Also P_omega_rel_dot formula like in your C++:
    # P_omega_rel_dot_i = P_omega_dot_P_i - P_omega_P_i x P_omega_B_i - R_PB * B_omega_dot_B
    P_omega_rel_dot = []
    for i in range(4):
        R_WP = kin["R_WP"][i]
        R_PW = R_WP.T
        R_PB = kin["R_PB"][i]
        P_omega_P_i = kin["P_omega_P"][i]
        P_omega_B_i = R_PB * wB
        P_omega_dot_P_i = R_PW * W_omega_dot_P[i]
        P_omega_rel_dot_i = P_omega_dot_P_i - P_omega_P_i.cross(P_omega_B_i) - (R_PB * B_omega_dot_B)
        P_omega_rel_dot.append(P_omega_rel_dot_i)

    # 2) Build f(x)
    f = sp.Matrix.zeros(kStateSize, 1)

    # position derivative
    f[0:3, 0] = v

    # base quaternion derivative
    f[3:7, 0] = quat_omega_dot(qB, wB)  # note: check frame consistency later

    # linear velocity derivative
    f[7:10, 0] = W_a_B

    # base angular velocity derivative (in body frame)
    f[10:13, 0] = B_omega_dot_B

    # arm quaternion derivatives
    for i in range(4):
        f[13+4*i : 17+4*i, 0] = quat_omega_dot(qA[i], omega_rel[i])

    # relative joint angular acceleration
    for i in range(4):
        f[29+3*i : 32+3*i, 0] = P_omega_rel_dot[i]

    return f

############################################################
# 6. YAML PARAMS (hard-coded from model/drone_parameters.yaml)
############################################################

def pose7_to_matrix(pose):
    """Convert [x,y,z,qw,qx,qy,qz] to 4x4 homogeneous matrix."""
    x_, y_, z_, qw, qx_, qy_, qz_ = pose
    qnorm = sp.sqrt(qw**2 + qx_**2 + qy_**2 + qz_**2)
    qw, qx_, qy_, qz_ = [v / qnorm for v in (qw, qx_, qy_, qz_)]
    R = quat_to_rot(sp.Matrix([qw, qx_, qy_, qz_]))
    T = sp.eye(4)
    T[:3, :3] = R
    T[:3, 3] = sp.Matrix([x_, y_, z_])
    return T

def params_from_yaml_constants():
    """Numeric params copied from model/drone_parameters.yaml."""
    # Masses / inertia
    mass_total = sp.Float("0.260")
    mass_base = sp.Float("0.157")
    mass_arm = (mass_total - mass_base) / 4.0
    I_arm = sp.Matrix.diag(sp.Float("0.0020"), sp.Float("0.0020"), sp.Float("0.0020"))
    I_base = sp.Matrix.diag(sp.Float("0.0020"), sp.Float("0.0020"), sp.Float("0.0020"))

    # Propellers / joints
    kappa_thrust_val = sp.Float("2.0e-7")
    kappa_torque_val = sp.Float("1.0e-10")
    rotor_inertia_val = sp.Float("1.0e-5")
    motor_dir_val = [1, 1, -1, -1]
    joint_damp_val = sp.Float("0.01")
    joint_stiff_val = sp.Float("1.0")

    # Transforms from YAML (qw,qx,qy,qz order)
    T_BH = [
        pose7_to_matrix([0.04, -0.032, 0.0, 0.92387953, 0.0, 0.0, -0.38268343]),
        pose7_to_matrix([-0.04, 0.032, 0.0, 0.38268343, 0.0, 0.0, 0.92387953]),
        pose7_to_matrix([0.04, 0.032, 0.0, 0.92387953, 0.0, 0.0, 0.38268343]),
        pose7_to_matrix([-0.04, -0.032, 0.0, 0.38268343, 0.0, 0.0, -0.92387953]),
    ]
    T_HP = [
        pose7_to_matrix([0.07, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        pose7_to_matrix([0.07, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        pose7_to_matrix([0.07, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        pose7_to_matrix([0.07, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    ]

    return {
        'mB': mass_base,
        'mP': mass_arm,
        'I_B': I_base,
        'I_A': I_arm,
        'thrust': thrust,
        'joint_damping': joint_damp_val,
        'joint_stiffness': joint_stiff_val,
        'kappa_thrust': kappa_thrust_val,
        'kappa_torque': kappa_torque_val,
        'rotor_inertia': rotor_inertia_val,
        'motor_dir': motor_dir_val,
        'T_BH': T_BH,
        'T_HP': T_HP,
    }

############################################################
# 7. JACOBIAN AND CODEGEN
############################################################

def main():
    print(">>> Loading parameters from YAML constants...")
    params = params_from_yaml_constants()

    print(">>> Building f(x) symbolically...")
    f_expr = build_f(x, params)

    print(">>> Computing Jacobian df/dx ... (this can take a while)")
    J_expr = f_expr.jacobian(x)

    print("State size:", kStateSize)
    print("f dimension:", f_expr.shape)
    print("Jacobian dimension:", J_expr.shape)

    print(">>> Generating C code for f and J ...")
    [(cf_name, cf_code),
     (hf_name, hf_code)] = codegen(
        name_expr=("drone_dynamics", f_expr),
        language="C",
        project="drone_dynamics",
        header=True
    )

    [(cJ_name, cJ_code),
     (hJ_name, hJ_code)] = codegen(
        name_expr=("drone_dynamics_jacobian", J_expr),
        language="C",
        project="drone_dynamics",
        header=True
    )

    # Write files
    with open(cf_name, 'w') as f:
        f.write(cf_code)
    with open(hf_name, 'w') as f:
        f.write(hf_code)
    with open(cJ_name, 'w') as f:
        f.write(cJ_code)
    with open(hJ_name, 'w') as f:
        f.write(hJ_code)

    print("Generated:", cf_name, hf_name, cJ_name, hJ_name)
    print(">>> Done.")

if __name__ == "__main__":
    main()
