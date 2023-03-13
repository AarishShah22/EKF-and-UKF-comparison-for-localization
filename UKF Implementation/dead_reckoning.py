import numpy as np
import matplotlib.pyplot as plt
import pickle

# Data Handling: Loading all data from .dat files to np arrays
lm_file_path = r"C:\Users\aarishs\Downloads\XYZ\Implementation 2\ds0_Landmark_Groundtruth.dat"
truth_file_path = r"C:\Users\aarishs\Downloads\XYZ\Implementation 2\ds0_Groundtruth.dat"
barcodes_file_path = r"C:\Users\aarishs\Downloads\XYZ\Implementation 2\ds0_Barcodes.dat"
meas_file_path = r"C:\Users\aarishs\Downloads\XYZ\Implementation 2\ds0_Measurement.dat"
odom_file_path = r"C:\Users\aarishs\Downloads\XYZ\Implementation 2\ds0_Odometry.dat"

lm_pos = np.zeros((15,3))
lm_std = np.zeros((15,3))
i = 0
lm_file = open(lm_file_path, "r")
for line in lm_file:
    if not line.startswith("#"):
        values = line.split()
        lm_pos[i,:] = np.array([values[0],values[1],values[2]])
        lm_std[i,:] = np.array([values[0],values[3], values[4]])
        i = i+1
lm_file.close()

l = 0
robot_true_pose = np.zeros((87676,4))
truth_file = open(truth_file_path, "r")
for line in truth_file:
    if not line.startswith("#"):
        values = line.split()
        robot_true_pose[l,:] = np.array([[values[0],values[1], values[2], values[3]]])
        l = l+1

truth_file.close()

k = 0
barcodes = np.zeros((20,2))
barcodes_file = open(barcodes_file_path, "r")
for line in barcodes_file:
    if not line.startswith("#"):
        values = line.split()
        barcodes[k,:] = np.array([[values[0], values[1]]])
        k = k+1

valid_barcodes = barcodes[5:,:]

barcodes_file.close()

m = 0
measurements = np.zeros((7720,4))
meas_file = open(meas_file_path, "r")
for line in meas_file:
    if not line.startswith("#"):
        values = line.split()
        measurements[m,:] = np.array([[values[0],values[1], values[2], values[3]]])
        m = m+1

meas_file.close()

n = 0
odom = np.zeros((95818,3))
odom_file = open(odom_file_path, "r")
for line in odom_file:
    if not line.startswith("#"):
        values = line.split()
        odom[n,:] = np.array([[values[0],values[1], values[2]]])
        n = n+1

odom_file.close()

# Observing path of robot based on just odometry
# measurements - "dead reckoning"
dead_reckon_pose = np.zeros((95818,4))
dead_reckon_pose[0,0] = odom[0,0]
dead_reckon_pose[0,1:] = robot_true_pose[0,1:]
noise = 0

for i in range(1,len(dead_reckon_pose)):
    
    t0 = dead_reckon_pose[i-1,0]
    x0 = dead_reckon_pose[i-1,1]
    y0 = dead_reckon_pose[i-1,2]
    theta0 = dead_reckon_pose[i-1,3]

    t1 = odom[i-1,0]
    v = odom[i-1,1]
    w = odom[i-1,2]

    dt = t1-t0
    dx = v*np.cos(theta0)*dt
    dy = v*np.sin(theta0)*dt
    dtheta = w*dt

    x1 = x0 + dx + np.random.randn()*noise
    y1 = y0 + dy + np.random.randn()*noise
    theta1 = theta0 + dtheta + np.random.randn()*noise
    dead_reckon_pose[i,0] = t1
    
    dead_reckon_pose[i,1:] = np.array([x1, y1, theta1])

dr_file_name = "dead_reckon_data"
dr_file_handle = open(dr_file_name,"wb")
pickle.dump(dead_reckon_pose,dr_file_handle)
dr_file_handle.close()