#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
機能:
- ユーザー手 (palm_pose) に追従（主タスク：DLS IK）
- ボトルのプレグラスプ姿勢を常時生成し、距離に応じて手→プレグラスプへ滑らかに吸着
- ヌル空間で 可操作性↑ + 関節リミット余裕↑ + プレグラスプ関節θ_preへの弱い引き戻し
- 「近距離で姿勢が悪い」時に、いったん後退→姿勢再整列→再接近の FSM を実行

FMS
"""

import math
import numpy as np
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray, String, Float64MultiArray

# ================== 設定 / 定数 =================================================================
arm_1_topic_name = "/forward_position_controller/commands"
arm_1_msg_type = Float64MultiArray

AMIR_ARM_BASE_Z = 0.316
AMIR_JOINT2_Z   = 0.097 
DEFAULT_L = [0.31, 0.31, 0.120]
AMIR_Q_INIT = [0.0, 0.0, 0.0, 0.0, 0.0]

# 把持向け yaw ロック（簡易）
YAW_LOCK_DEG = 0
EE_PITCH_LOCK_DEG = 0.0


# IK/到達性関連
SIGMA_MIN_THR = 0.03   # 最小特異値閾値
COND_MAX_THR  = 60.0   # 条件数閾値
ERR_THRESH    = 0.01   # 収束誤差閾値 (r,z,phiの二乗和)

# ヌル空間関連
W_MANI  = 1.0          # 可操作性重み
W_LIMIT = 0.3          # 関節余裕重み
K_NULL  = 0.02         # ヌル空間ステップ

JOINT_LIMITS = [
    (0.05, 2.30),          # Joint_2: 物理限界 0.0〜2.356 より少し内側
    (-2.70, -0.05),        # Joint_3: 物理限界 -2.792〜0.0 より少し内側
    (-2.00, 1.20),         # Joint_4: 物理限界 -2.094〜1.309 より少し内側
]

# ブレンドと吸着挙動
PREGRASP_OFFSET = -0.09   # r方向の手前オフセット[m]
D_IN   = 0.18            # 吸着開始距離（以下で吸着）
D_OUT  = 0.3            # 吸着解除距離（以上で解除）
SIG_K  = 40.0            # シグモイド鋭さ
BETA_SMOOTH = 0.4        # βの滑らか化係数

# ===== 退避→再整列→再接近 FSM パラメータ ===========================================================
class Phase(Enum):
    TRACK   = 0   # 通常追従/プレグラスプ吸着
    BACKOFF = 1   # 後退（スタンドオフへ）
    REALIGN = 2   # 姿勢再整列（ヌル空間最適化を重視）
    APPROACH= 3   # 再接近（pregrasp/targetへ）

BACKOFF_D       = 0.05  # [m] 退避量（r方向に手前へ）
MARGIN_MIN_THR  = 0.08   # [rad] 最小マージンしきい値
REALIGN_DT_MAX  = 0.6    # [s] 再整列の最長時間（フェイルセーフ）
IMPROVE_H_MIN   = 0.08   # H 改善の目安
CD_SWITCH       = 0.5    # [s] 状態切替のクールダウン
APPROACH_LOCK_BETA = True  # 再接近時は beta≈1.0 に固定（向きの安定化）

# ================== ユーティリティ =================================================================
def DegToRad(th): return (np.pi/180.0)*th
def RadToDeg(th): return (180.0/np.pi)*th

def angle_wrap(a):
    # [-pi, pi) 正規化
    return (a + math.pi) % (2*math.pi) - math.pi

def Theta0(x,y):
    if x == 0 and y == 0:
        return 0.0
    return math.atan2(y,x)

def make_arm_msg(arm_js):
    """
    arm_js: [Joint_1, Joint_2, Joint_3, Joint_4, Joint_5] [rad]
    """
    return Float64MultiArray(data=[float(v) for v in arm_js])

def joint_margins(th, limits):
    m = []
    for i,(lo,hi) in enumerate(limits):
        m.append(min(th[i]-lo, hi-th[i]))
    return m  # [m0,m1,m2]

def e_task_norm(th, target, L):
    cur = fk(L, th[0], th[1], th[2])
    dr = target[0]-cur[0]
    dz = target[1]-cur[1]
    dphi = angle_wrap(target[2]-cur[2])
    return math.sqrt(dr*dr + dz*dz + dphi*dphi)

def cond_and_sigma(J):
    U,S,Vt = np.linalg.svd(J, full_matrices=False)
    smin = float(S.min())
    cond = float(S.max()/S.min()) if S.min() > 1e-8 else float('inf')
    return cond, smin

# ---- 運動学（等価3リンク: (r,z,phi) に対応） ----
def fk(L, th0, th1, th2):
    # AMIR planar FK from arm base link0_1
    # th0, th1, th2 = Joint_2, Joint_3, Joint_4
    # r: forward direction, z: upward direction

    a0 = th0
    a1 = th0 + th1
    a2 = th0 + th1 + th2

    r = (
        L[0] * math.cos(a0)
        + L[1] * math.cos(a1)
        + L[2] * math.cos(a2)
    )

    z = (
        AMIR_JOINT2_Z
        + L[0] * math.sin(a0)
        + L[1] * math.sin(a1)
        + L[2] * math.sin(a2)
    )

    phi = a2
    return [r, z, phi]


def jacobian(L, th0, th1, th2):
    a0 = th0
    a1 = th0 + th1
    a2 = th0 + th1 + th2

    # dr / dtheta
    r11 = -L[0] * math.sin(a0) - L[1] * math.sin(a1) - L[2] * math.sin(a2)
    r12 = -L[1] * math.sin(a1) - L[2] * math.sin(a2)
    r13 = -L[2] * math.sin(a2)

    # dz / dtheta
    r21 = L[0] * math.cos(a0) + L[1] * math.cos(a1) + L[2] * math.cos(a2)
    r22 = L[1] * math.cos(a1) + L[2] * math.cos(a2)
    r23 = L[2] * math.cos(a2)

    # dphi / dtheta
    r31 = 1.0
    r32 = 1.0
    r33 = 1.0

    return np.array([
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33]
    ], dtype=np.float64)

def velocity(pd, p):
    # 簡易P制御: (r,z,phi)
    k = 0.1
    vr = k * (pd[0] - p[0])
    vz = k * (pd[1] - p[1])
    w  = k * angle_wrap(pd[2] - p[2])
    return np.array([[vr],[vz],[w]], dtype=np.float64)

def inversekinematics(L, pd, p0, lam):
    # DLS一歩分（増分Δθ）
    p_fk = fk(L, p0[0], p0[1], p0[2])
    v = velocity(pd, p_fk)
    J = jacobian(L, p0[0], p0[1], p0[2])
    I = np.identity(3)
    JT = J.T
    dtheta = (np.linalg.pinv(JT @ J + lam*I) @ JT) @ v
    return 0.1 * dtheta  # ステップスケール

def dls_step(L, target, th, lam=1.0):
    cur = fk(L, th[0], th[1], th[2])
    err = np.array([[target[0]-cur[0]], [target[1]-cur[1]], [angle_wrap(target[2]-cur[2])]], dtype=np.float64)
    J = jacobian(L, th[0], th[1], th[2])
    JJt = J @ J.T
    J_pinv = J.T @ np.linalg.inv(JJt + lam*np.eye(3))
    dth = (J_pinv @ err).reshape(3)
    return dth, err, J

def ik_converges(L, target, th0, max_iters=200):
    th = th0.copy()
    lam = 1.0
    J = jacobian(L, th[0], th[1], th[2])
    err = np.zeros((3,1))
    for _ in range(max_iters):
        dth, err, J = dls_step(L, target, th, lam)
        e = float(err.T @ err)
        lam = e + 0.002
        th += dth
        if e < ERR_THRESH**2:
            U,S,Vt = np.linalg.svd(J, full_matrices=False)
            smin = S.min()
            cond = (S.max()/S.min()) if S.min()>1e-8 else float('inf')
            return True, e, smin, cond, th
    U,S,Vt = np.linalg.svd(J, full_matrices=False)
    smin = S.min()
    cond = (S.max()/S.min()) if S.min()>1e-8 else float('inf')
    return False, float(err.T @ err), smin, cond, th

def manipulability(J):
    JJt = J @ J.T
    det = np.linalg.det(JJt)
    if det <= 1e-12: return -1e6
    return math.log(det)

def limit_margin_cost(th, limits=JOINT_LIMITS, eps=1e-3):
    s = 0.0
    for i,(lo,hi) in enumerate(limits):
        m = min(th[i]-lo, hi-th[i])
        m = max(m, eps)
        s += math.log(m)
    return s

def merit_function(th, L):
    J = jacobian(L, th[0], th[1], th[2])
    return W_MANI * manipulability(J) + W_LIMIT * limit_margin_cost(th)

def finite_diff_grad(f, th, L, h=1e-3):
    #勾配計算
    g = np.zeros(3, dtype=np.float64)
    base = f(th, L)
    for i in range(3):
        th_p = th.copy(); th_p[i] += h
        g[i] = (f(th_p, L) - base)/h
    return g

def nullspace_projector(J):
    JJt = J @ J.T
    J_pinv = J.T @ np.linalg.inv(JJt + 1e-6*np.eye(3))
    I = np.eye(3)
    return I - J_pinv @ J

# ===== プレグラスプ生成 & ブレンド =================================================================
def make_pregrasp_from_bottle(bxyz, approach_yaw_deg=YAW_LOCK_DEG, offset=PREGRASP_OFFSET):
    bx, by, bz = bxyz
    r = math.sqrt(bx*bx + by*by)
    yaw = DegToRad(approach_yaw_deg)
    r_pre = max(r - offset, 0.02)

    # base_footprint基準 → arm base基準
    return [r_pre, bz - AMIR_ARM_BASE_Z, yaw]

_beta = 0.0
_near = False
def blend_weight(distance, d_in=D_IN, d_out=D_OUT, k=SIG_K, alpha=BETA_SMOOTH):
    """ヒステリシス付きスムーズβ"""
    global _beta, _near
    if _near:
        if distance >= d_out: _near = False
    else:
        if distance <= d_in: _near = True
    d0 = d_in if _near else d_out
    b_inst = 1.0 / (1.0 + math.exp(k*(distance - d0)))
    _beta = (1-alpha)*_beta + alpha*b_inst
    return _beta

def merit_pregrasp_aug(th, L, th_pre=None):
    base = merit_function(th, L)
    extra = 0.0
    if th_pre is not None:
        #Pregraspがあったら
        extra += 0.3 * (- np.sum((th - th_pre)**2))  # 弱い引き戻し
    return base + extra

# ================== 購読コールバック =========================================================================
palm_pose   = PoseStamped()
bottle_xyz  = np.zeros(3, dtype=np.float64)

def palm_cb(msg):
    global palm_pose
    palm_pose = msg

def bottle_cb(msg):
    """Float32MultiArray: data=[x,y,z,...] を想定"""
    global bottle_xyz
    if len(msg.data) >= 3:
        bottle_xyz[0] = -float(msg.data[0])
        bottle_xyz[1] = -float(msg.data[2])
        bottle_xyz[2] = float(msg.data[1])

def dls_step_rz(L, target, th, lam=0.01):
    cur = fk(L, th[0], th[1], th[2])

    err = np.array([
        [target[0] - cur[0]],
        [target[1] - cur[1]]
    ], dtype=np.float64)

    J_full = jacobian(L, th[0], th[1], th[2])
    J = J_full[:2, :]  # r,zだけ使う

    JJt = J @ J.T
    J_pinv = J.T @ np.linalg.inv(JJt + lam * np.eye(2))

    dth = (J_pinv @ err).reshape(3)
    return dth, err, J


# ================== メイン ========================================================================================================
class AmirTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('amir_trajectory_publisher')

        # ==================
        # Parameters
        # ==================
        self.declare_parameter("L1", DEFAULT_L[0])
        self.declare_parameter("L2", DEFAULT_L[1])
        self.declare_parameter("L3", DEFAULT_L[2])

        self.declare_parameter("th1_init", 3.2)    # Joint_2
        self.declare_parameter("th2_init", -2.2)   # Joint_3
        self.declare_parameter("th3_init", 1.0)    # Joint_4

        self.declare_parameter("alpha", 0.05)
        self.declare_parameter("k_null", K_NULL)

        self.L = [
            self.get_parameter("L1").value,
            self.get_parameter("L2").value,
            self.get_parameter("L3").value,
        ]

        self.theta0 = self.get_parameter("th1_init").value
        self.theta1 = self.get_parameter("th2_init").value
        self.theta2 = self.get_parameter("th3_init").value

        self.alpha = self.get_parameter("alpha").value
        self.k_null = self.get_parameter("k_null").value

        # ==================
        # State
        # ==================
        self.palm_pose = PoseStamped()
        self.bottle_xyz = np.zeros(3, dtype=np.float64)

        self.phase = Phase.TRACK
        self.last_switch_t = 0.0
        self.realign_start_t = 0.0
        self.H_baseline = None
        self.stand_rzp = None

        self.t0 = self.get_clock().now()

        # ==================
        # Subscribers
        # ==================
        self.create_subscription(
            PoseStamped,
            "palm_pose",
            self.palm_cb,
            10
        )

        self.create_subscription(
            Float32MultiArray,
            "/identified_bottle_pose",
            self.bottle_cb,
            10
        )

        # ==================
        # Publishers
        # ==================
        self.arm_cmd_pub = self.create_publisher(
            Float64MultiArray,
            arm_1_topic_name,
            5
        )

        self.joint_pub_deg = self.create_publisher(
            Float32MultiArray,
            "/arm_joint_command",
            10
        )

        self.metrics_pub = self.create_publisher(
            Float32MultiArray,
            "/ybt_metrics",
            20
        )

        # ROS1の latch=True 相当
        latch_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.phase_names_pub = self.create_publisher(
            String,
            "/phase_name",
            latch_qos
        )

        self.names_pub = self.create_publisher(
            String,
            "/ybt_metric_names",
            latch_qos
        )

        self.names_pub.publish(String(
            data="t,dist,beta,e_task,sigma_min,cond,mani,margin0,margin1,margin2,H,dH_null,Jdnull_norm,dnull_norm,de_due_null,phase"
        ))

        self.get_logger().info("Start AMIR trajectory publisher for ROS2")

        # 30 Hz
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)

    def palm_cb(self, msg):
        self.palm_pose = msg

    def bottle_cb(self, msg):
        """
        Float32MultiArray: data=[x,y,z,...] を想定
        Unity/ROS座標変換をここで実施
        """
        if len(msg.data) >= 3:
            self.bottle_xyz[0] = -float(msg.data[0])
            self.bottle_xyz[1] = -float(msg.data[2])
            self.bottle_xyz[2] =  float(msg.data[1])

    def timer_callback(self):
        now = (self.get_clock().now() - self.t0).nanoseconds * 1e-9

        theta0 = self.theta0
        theta1 = self.theta1
        theta2 = self.theta2

        # ==================
        # 手ポーズ取得
        # ==================
        if (
            self.palm_pose.pose.position.x == 0 and
            self.palm_pose.pose.position.y == 0 and
            self.palm_pose.pose.position.z == 0
        ):
            pose_holo = [0.3, 0.0, AMIR_ARM_BASE_Z + 0.2]
        else:
            pose_holo = [
                self.palm_pose.pose.position.x,
                self.palm_pose.pose.position.y,
                self.palm_pose.pose.position.z,
            ]

        joint1 = math.atan2(pose_holo[1], pose_holo[0])

        r_palm = math.sqrt(pose_holo[0] ** 2 + pose_holo[1] ** 2)
        z_palm = pose_holo[2] - AMIR_ARM_BASE_Z
        phi_palm = DegToRad(YAW_LOCK_DEG)

        palm_rzp = [r_palm, z_palm, phi_palm]

        # ==================
        # プレグラスプ生成 & ブレンド
        # ==================
        if np.linalg.norm(self.bottle_xyz) > 1e-6:
            pre_rzp = make_pregrasp_from_bottle(self.bottle_xyz)

            dist = math.sqrt(
                (pose_holo[0] - self.bottle_xyz[0]) ** 2 +
                (pose_holo[1] - self.bottle_xyz[1]) ** 2
            )

            beta = blend_weight(dist)
        else:
            pre_rzp = palm_rzp
            beta = 0.0
            dist = -1.0

        if (
            APPROACH_LOCK_BETA and
            self.phase == Phase.APPROACH and
            np.linalg.norm(self.bottle_xyz) > 1e-6
        ):
            beta = 1.0

        target_rzp = [
            (1 - beta) * palm_rzp[0] + beta * pre_rzp[0],
            (1 - beta) * palm_rzp[1] + beta * pre_rzp[1],
            (1 - beta) * palm_rzp[2] + beta * pre_rzp[2],
        ]

        MAX_R = 0.68
        MIN_R = 0.08
        target_rzp[0] = float(np.clip(target_rzp[0], MIN_R, MAX_R))

        # ==================
        # プレグラスプ関節 θ_pre
        # ==================
        th_pre = None
        if np.linalg.norm(self.bottle_xyz) > 1e-6:
            ok_pg, _, _, _, th_try = ik_converges(
                self.L,
                pre_rzp,
                np.array([theta0, theta1, theta2], dtype=np.float64)
            )
            if ok_pg:
                th_pre = th_try

        # ==================
        # 主タスク DLS IK
        # ==================
        lam = 0.01

        for _ in range(50):
            dth, err, J_task = dls_step(
                self.L,
                target_rzp,
                np.array([theta0, theta1, theta2], dtype=np.float64),
                lam=lam
            )

            e = float(err.T @ err)
            lam = max(e + 0.002, 0.002)

            theta0 += float(self.alpha * dth[0])
            theta1 += float(self.alpha * dth[1])
            theta2 += float(self.alpha * dth[2])

            if e < 1e-6:
                break

        # ==================
        # ヌル空間更新
        # ==================
        th_vec_before = np.array([theta0, theta1, theta2], dtype=np.float64)
        e_before = e_task_norm(th_vec_before, target_rzp, self.L)

        J_now = jacobian(self.L, theta0, theta1, theta2)
        cond_now, sigma_min_now = cond_and_sigma(J_now)

        mani_now = manipulability(J_now)
        m0, m1, m2 = joint_margins(th_vec_before, JOINT_LIMITS)
        H_before = merit_pregrasp_aug(th_vec_before, self.L, th_pre)

        P = nullspace_projector(J_now)
        gradH = finite_diff_grad(
            lambda th, L_: merit_pregrasp_aug(th, L_, th_pre),
            th_vec_before,
            self.L
        )

        dth_null = self.k_null * (P @ gradH)

        theta0 += float(dth_null[0])
        theta1 += float(dth_null[1])
        theta2 += float(dth_null[2])

        # ==================
        # Joint limit clip
        # ==================
        theta0 = float(np.clip(theta0, JOINT_LIMITS[0][0], JOINT_LIMITS[0][1]))
        theta1 = float(np.clip(theta1, JOINT_LIMITS[1][0], JOINT_LIMITS[1][1]))
        theta2 = float(np.clip(theta2, JOINT_LIMITS[2][0], JOINT_LIMITS[2][1]))

        th_vec_after = np.array([theta0, theta1, theta2], dtype=np.float64)

        H_after = merit_pregrasp_aug(th_vec_after, self.L, th_pre)
        dH_null = H_after - H_before

        e_after = e_task_norm(th_vec_after, target_rzp, self.L)
        de_due_null = e_after - e_before

        Jdnull = J_now @ dth_null.reshape(3, 1)
        Jdnull_norm = float(np.linalg.norm(Jdnull))
        dnull_norm = float(np.linalg.norm(dth_null))

        # ==================
        # FSM判定
        # ==================
        cooldown_ok = (now - self.last_switch_t) > CD_SWITCH

        near = (
            np.linalg.norm(self.bottle_xyz) > 1e-6 and
            dist >= 0.0 and
            dist <= D_IN + 0.03
        )

        m_min = min(m0, m1, m2)

        bad_posture = (
            sigma_min_now < SIGMA_MIN_THR or
            cond_now > COND_MAX_THR or
            m_min < MARGIN_MIN_THR
        )

        phase_name = self.phase.name

        if self.phase == Phase.TRACK:
            phase_name = "TRACK"

            if cooldown_ok and near and bad_posture:
                r_stand = pre_rzp[0] + BACKOFF_D
                self.stand_rzp = [r_stand, pre_rzp[1], pre_rzp[2]]

                # 注意:
                # 元コードでは phase = Phase.BACKOFF がコメントアウトされていた。
                # 本当にFSMを動かすなら下を有効化する。
                # self.phase = Phase.BACKOFF

                self.last_switch_t = now
                phase_name = "BACKOFF"

        # ==================
        # Publish
        # ==================
        cand = [
            joint1,
            theta0,
            theta1,
            theta2,
            0.0
        ]

        cand_deg = [RadToDeg(c) for c in cand]

        metrics = [
            float(now),
            float(dist),
            float(beta),
            float(e_after),
            float(sigma_min_now),
            float(cond_now),
            float(mani_now),
            float(m0),
            float(m1),
            float(m2),
            float(H_after),
            float(dH_null),
            float(Jdnull_norm),
            float(dnull_norm),
            float(de_due_null),
            float(self.phase.value),
        ]

        self.phase_names_pub.publish(String(data=phase_name))
        self.metrics_pub.publish(Float32MultiArray(data=metrics))
        self.joint_pub_deg.publish(Float32MultiArray(data=cand_deg))
        self.arm_cmd_pub.publish(make_arm_msg(cand))

        # 状態保存
        self.theta0 = theta0
        self.theta1 = theta1
        self.theta2 = theta2


def main(args=None):
    rclpy.init(args=args)

    node = AmirTrajectoryPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()