import mujoco
import mujoco.viewer
import numpy as np
import time
import os

# Pfad zur MJCF-XML-Datei (ersetzen durch deinen tatsächlichen Pfad)
XML_PATH = "scene.xml"  # z. B. "../models/h1.xml"

# Lade das Modell
model = mujoco.MjModel.from_xml_path(XML_PATH)
h1 = mujoco.MjModel.from_xml_path("h1.xml")

# Erstelle ein Data-Objekt (Simulation)
data = mujoco.MjData(model)

key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
qpos_ref = model.key_qpos[key_id].copy()

#qpos_ref = h1.key_qpos[h1.key_name2id("home")]

kp = 1500.0
kd = 50.0

# Mapping von Jointnamen zu Actuator-IDs
actuator_ids = {model.actuator(i).name: i for i in range(model.nu)}


act_id_left_hip_yaw_joint_desired = actuator_ids["left_hip_yaw_joint"]  
act_id_left_hip_roll_joint_desired = actuator_ids["left_hip_roll_joint"]  
act_id_left_hip_pitch_joint_desired = actuator_ids["left_hip_pitch_joint"]  
act_id_left_knee_joint_desired = actuator_ids["left_knee_joint"]  
act_id_left_ankle_joint_desired = actuator_ids["left_ankle_joint"]  
act_id_right_hip_yaw_joint_desired = actuator_ids["right_hip_yaw_joint"]  
act_id_right_hip_roll_joint_desired = actuator_ids["right_hip_roll_joint"]  
act_id_right_hip_pitch_joint_desired = actuator_ids["right_hip_pitch_joint"]  
act_id_right_knee_joint_desired = actuator_ids["right_knee_joint"]  
act_id_right_ankle_joint_desired = actuator_ids["right_ankle_joint"]  
act_id_torso_joint_desired = actuator_ids["torso_joint"]  
act_id_left_shoulder_pitch_joint_desired = actuator_ids["left_shoulder_pitch_joint"]  
act_id_left_shoulder_roll_joint_desired = actuator_ids["left_shoulder_roll_joint"]  
act_id_left_shoulder_yaw_joint_desired = actuator_ids["left_shoulder_yaw_joint"]  
act_id_left_elbow_joint_desired = actuator_ids["left_elbow_joint"] 
act_id_right_shoulder_pitch_joint_desired = actuator_ids["right_shoulder_pitch_joint"]  
act_id_right_shoulder_roll_joint_desired = actuator_ids["right_shoulder_roll_joint"] 
act_id_right_shoulder_yaw_joint_desired = actuator_ids["right_shoulder_yaw_joint"]  
act_id_right_elbow_joint_desired = actuator_ids["right_elbow_joint"]  


def compute_center_of_mass(model, data):
    total_mass = 0.0
    weighted_pos_sum = np.zeros(3)

    for i in range(model.nbody):
        mass = model.body_mass[i]
        pos = data.xipos[i]  #world coordinated of the center of the body
        weighted_pos_sum += mass * pos
        total_mass += mass

    com = weighted_pos_sum / total_mass
    return com

def compute_com_velocity(model, data):
    total_mass = 0.0
    weighted_vel_sum = np.zeros(3)

    for i in range(model.nbody):
        mass = model.body_mass[i]
        vel = data.cvel[i][:3]  # linear velocity
        weighted_vel_sum += mass * vel
        total_mass += mass

    com_vel = weighted_vel_sum / total_mass
    return com_vel


def arm_controll(data, t):     
    q_error_left_shoulder_roll = qpos_ref[19] - data.qpos[19]
    qvel_error_left_shoulder_roll = -data.qvel[18]
    data.ctrl[act_id_left_shoulder_roll_joint_desired] = kp * q_error_left_shoulder_roll + kd * qvel_error_left_shoulder_roll

    q_error_left_shoulder_yaw = qpos_ref[20] - data.qpos[20]
    qvel_error_left_shoulder_yaw = -data.qvel[19]
    data.ctrl[act_id_left_shoulder_yaw_joint_desired] = kp * q_error_left_shoulder_yaw + kd * qvel_error_left_shoulder_yaw

    q_error_left_elbow = qpos_ref[21] - data.qpos[21]
    qvel_error_left_elbow = -data.qvel[20]
    data.ctrl[act_id_left_elbow_joint_desired] = kp * q_error_left_elbow + kd * qvel_error_left_elbow

    q_error_right_shoulder_roll = qpos_ref[23] - data.qpos[23]
    qvel_error_right_shoulder_roll = -data.qvel[22]
    data.ctrl[act_id_right_shoulder_roll_joint_desired] = kp * q_error_right_shoulder_roll + kd * qvel_error_right_shoulder_roll

    q_error_right_shoulder_yaw = qpos_ref[24] - data.qpos[24]
    qvel_error_right_shoulder_yaw = -data.qvel[23]
    data.ctrl[act_id_right_shoulder_yaw_joint_desired] = kp * q_error_right_shoulder_yaw + kd * qvel_error_right_shoulder_yaw

    q_error_right_elbow = qpos_ref[25] - data.qpos[25]
    qvel_error_right_elbow = -data.qvel[24]
    data.ctrl[act_id_right_elbow_joint_desired] = kp * q_error_right_elbow + kd * qvel_error_right_elbow

        
def leg_control(data, t):
    q_error_left_hip_yaw = qpos_ref[7] - data.qpos[7] 
    qvel_error_left__hip_yaw = -data.qvel[6]
    data.ctrl[act_id_left_hip_yaw_joint_desired] = kp * (q_error_left_hip_yaw) + kd * qvel_error_left__hip_yaw

    q_error_left_hip_roll = qpos_ref[8] - data.qpos[8] 
    qvel_error_left__hip_roll = -data.qvel[7]
    data.ctrl[act_id_left_hip_roll_joint_desired] = kp * (q_error_left_hip_roll) + kd * qvel_error_left__hip_roll

    q_error_left_hip_pitch = qpos_ref[9] - data.qpos[9] 
    qvel_error_left__hip_pitch = -data.qvel[8]
    data.ctrl[act_id_left_hip_pitch_joint_desired] = kp * (q_error_left_hip_pitch) + kd * qvel_error_left__hip_pitch

    q_error_left_knee = qpos_ref[10] - data.qpos[10] 
    qvel_error_left_knee = -data.qvel[9]
    data.ctrl[act_id_left_knee_joint_desired] = kp * (q_error_left_knee) + kd * qvel_error_left_knee

    q_error_left_ankle = qpos_ref[11] - data.qpos[11] 
    qvel_error_left_ankle = -data.qvel[10]
    data.ctrl[act_id_left_ankle_joint_desired] = kp * (q_error_left_ankle) + kd * qvel_error_left_ankle


    q_error_right_hip_yaw = qpos_ref[12] - data.qpos[12] 
    qvel_error_right__hip_yaw = -data.qvel[11]
    data.ctrl[act_id_right_hip_yaw_joint_desired] = kp * (q_error_right_hip_yaw) + kd * qvel_error_right__hip_yaw

    q_error_rigth_hip_roll = qpos_ref[13] - data.qpos[13] 
    qvel_error_right__hip_roll = -data.qvel[12]
    data.ctrl[act_id_right_hip_roll_joint_desired] = kp * (q_error_rigth_hip_roll) + kd * qvel_error_right__hip_roll

    q_error_right_hip_pitch = qpos_ref[14] - data.qpos[14] 
    qvel_error_rigth__hip_pitch = -data.qvel[13]
    data.ctrl[act_id_right_hip_pitch_joint_desired] = kp * (q_error_right_hip_pitch) + kd * qvel_error_rigth__hip_pitch

    q_error_right_knee = qpos_ref[15] - data.qpos[15] 
    qvel_error_right_knee = -data.qvel[14]
    data.ctrl[act_id_right_knee_joint_desired] = kp * (q_error_right_knee) + kd * qvel_error_right_knee

    q_error_right_ankle = qpos_ref[16] - data.qpos[16] 
    qvel_error_right_ankle = -data.qvel[15]
    data.ctrl[act_id_right_ankle_joint_desired] = kp * (q_error_right_ankle) + kd * qvel_error_right_ankle

    q_error_torso = qpos_ref[17] - data.qpos[17]
    qvel_error_torso = -data.qvel[16]
    data.ctrl[act_id_torso_joint_desired] = kp * (q_error_torso) + kd * qvel_error_torso

    #print("qpos:")
    #print(data.qpos[act_id_left_hip_yaw_joint_desired ]) #
    #print(data.qpos[act_id_left_hip_roll_joint_desired ]) #
    #print(data.qpos[act_id_left_hip_pitch_joint_desired])# 
    ##print(data.qpos[act_id_left_knee_joint_desired ])#  
    #print(data.qpos[act_id_left_ankle_joint_desired]) #
    #print(data.qpos[act_id_right_hip_yaw_joint_desired]) #
    #print(data.qpos[act_id_right_hip_roll_joint_desired ]) #    
    #print(data.qpos[act_id_right_hip_pitch_joint_desired ])# is left hip yaw            7
    #print(data.qpos[act_id_right_knee_joint_desired  ]) # is left hip roll              8
    ##print(data.qpos[act_id_right_ankle_joint_desired ]) #is left hip pich               9
    #print(data.qpos[act_id_torso_joint_desired]) # is left knee                        10
    #print(data.qpos[act_id_left_shoulder_pitch_joint_desired])  #is left ankle         11
    #print(data.qpos[act_id_left_shoulder_roll_joint_desired])  #is right hip yaw       12
    #print(data.qpos[act_id_left_shoulder_yaw_joint_desired])  #is right hip roll       13
    #print(data.qpos[act_id_left_elbow_joint_desired])  #right hip pitch                14
    #print(data.qpos[act_id_right_shoulder_pitch_joint_desired])  #is right knee        15
    #print(data.qpos[act_id_right_shoulder_roll_joint_desired])  #rigth ankle           16
    #print(data.qpos[17])                                                              #17
    #print(data.qpos[act_id_right_elbow_joint_desired]) #is left shoulder (1)           18
    #print(data.qpos[19])                                                              #19
    #print(data.qpos[20])                                                              #20
    #print(data.qpos[21])                                                              #21
    #print(data.qpos[22]) #is right shoulder (1)                                       #22
    #print(data.qpos[23])                                                              #23
    #print(data.qpos[24])                                                              #24
    #print(data.qpos[25])                                                              #25
    
def balance_control(model, data, t):
    center_of_mass = compute_center_of_mass(model, data)
    com_velocity = compute_com_velocity(model, data)

    #left shoulder pitch to stabelize center of mass
    q_error_left_shoulder_pitch = - (0.2 - 10* center_of_mass[0]) + qpos_ref[18] - data.qpos[18] 
    qvel_error_left_shoulder_pitch = - 0.9* com_velocity[1] -data.qvel[17]
    data.ctrl[act_id_left_shoulder_pitch_joint_desired] = 20 * q_error_left_shoulder_pitch + 5 * qvel_error_left_shoulder_pitch
    print("-----")
    #print( q_error_left_shoulder_pitch )
    #print( qvel_error_left_shoulder_pitch )
    #print(20 * q_error_left_shoulder_pitch + 2 * qvel_error_left_shoulder_pitch)
    print(center_of_mass)
    print(com_velocity)
   
    #right shoulder pitch to stabelize center of mass
    q_error_right_shoulder_pitch = - (0.2 - 10* center_of_mass[0]) + qpos_ref[22] - data.qpos[22]
    qvel_error_right_shoulder_pitch =  - 0.9* com_velocity[1] -data.qvel[21]
    data.ctrl[act_id_right_shoulder_pitch_joint_desired] = 20 * q_error_right_shoulder_pitch + 5 * qvel_error_right_shoulder_pitch

 
# start simulation with viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running():
        t = time.time() - start
        
        arm_controll(data, t)
        leg_control(data, t)
        balance_control(model, data, t)

        mujoco.mj_step(model, data)
        viewer.sync()
