"""
Sample camera configuration file for L2G to UniAD conversion.
This file contains the camera setup from l2g_data.md format.
"""

# Camera configuration matching l2g_data.md format
cams = {
    '100f': {  # Front camera
        'data_path': 'image/lucid_cameras_x00.gige_100_f_hdr.h265',
        'sensor2ego_translation': [0.005, -0.125, -0.171],
        'sensor2ego_rotation': [-0.525, 0.509, -0.491, -0.495],  # [w,x,y,z] quaternion
        'cam_intrinsic': [
            657.904403, 0.0, 714.717385597,
            0.0, 658.500765392, 463.1351298,
            0.0, 0.0, 1.0
        ]
    },
    '60b': {  # Back camera
        'data_path': 'image/lucid_cameras_x00.gige_60_b_hdr.h265',
        'sensor2ego_translation': [-0.011, -0.034, -2.469],
        'sensor2ego_rotation': [0.508, 0.513, -0.486, 0.493],  # [w,x,y,z] quaternion
        'cam_intrinsic': [
            1033.063670, 0.000000, 730.505500,
            0.000000, 1034.542542, 468.282976,
            0.000000, 0.000000, 1.000000
        ]
    },
    '100fl': {  # Front left camera
        'data_path': 'image/lucid_cameras_x01.gige_100_fl_hdr.h265',
        'sensor2ego_translation': [0.213, -0.073, -0.671],
        'sensor2ego_rotation': [0.691, -0.169, 0.162, 0.684],  # [w,x,y,z] quaternion
        'cam_intrinsic': [
            660.05027892, 0.0, 720.216502034,
            0.0, 660.349325206, 467.66874548,
            0.0, 0.0, 1.0
        ]
    },
    '100fr': {  # Front right camera
        'data_path': 'image/lucid_cameras_x01.gige_100_fr_hdr.h265',
        'sensor2ego_translation': [-0.223, -0.081, -0.654],
        'sensor2ego_rotation': [-0.177, 0.688, -0.682, -0.176],  # [w,x,y,z] quaternion
        'cam_intrinsic': [
            659.20444086, 0.0, 713.833685513,
            0.0, 659.864183739, 455.950517493,
            0.0, 0.0, 1.0
        ]
    }
}

# Note: ego2global_translation and ego2global_rotation will be extracted from canbus data
# Format from canbus:
# - ego2global_translation = canbus data index 0-2: position_xyz (x, y, z coordinates)
# - ego2global_rotation = canbus data index 3-6: quaternion_wxyz