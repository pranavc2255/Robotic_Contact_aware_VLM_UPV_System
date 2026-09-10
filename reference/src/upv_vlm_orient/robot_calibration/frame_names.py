"""Named coordinate frames used by the UPV robot calibration layer."""

BASE = "base"
TCP = "tcp"
CAMERA_COLOR_OPTICAL = "camera_color_optical"
CAMERA_DEPTH_OPTICAL = "camera_depth_optical"
UPV_TOOL_BODY = "upv_tool_body"
UPV_CONTACT_MIDPLANE = "upv_contact_midplane"
UPV_LEFT_CONTACT_CENTER = "upv_left_contact_center"
UPV_RIGHT_CONTACT_CENTER = "upv_right_contact_center"
TARGET_OBJECT = "target_object"
TARGET_CONTACT = "target_contact"

FRAME_DESCRIPTIONS = {
    BASE: "UR robot base frame.",
    TCP: "Active RTDE TCP/tool pose frame used by moveL/getActualTCPPose.",
    CAMERA_COLOR_OPTICAL: "RealSense RGB optical frame where pixels/depth are measured.",
    CAMERA_DEPTH_OPTICAL: "RealSense depth optical frame/aligned depth frame.",
    UPV_TOOL_BODY: "Rigid body frame for the UPV end-effector assembly.",
    UPV_CONTACT_MIDPLANE: "Midpoint frame between two UPV transducer contact faces/jaws.",
    UPV_LEFT_CONTACT_CENTER: "Left transducer contact face center.",
    UPV_RIGHT_CONTACT_CENTER: "Right transducer contact face center.",
    TARGET_OBJECT: "Vision-derived object frame with origin at mask centroid and x-axis along major axis.",
    TARGET_CONTACT: "Desired contact/measurement frame on the object.",
}

