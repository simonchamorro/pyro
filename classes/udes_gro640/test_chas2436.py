#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26

@author: Simon Chamorro CHAS2436
------------------------------------


Tests for GRO640 

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gro640_robots import *
from chas2436      import *



def compare_arrays(a, b, e=0.001):
    equal = True
    for i in range(len(a)):
        if (abs(a[i] - b[i]) > e):
            equal = False
    return equal


def plot_kuka(points1, points2):
    gs = gridspec.GridSpec(1, 2)
    fig = plt.figure()
    fig.suptitle("Cinématique directe du Kuka")

    ax = plt.subplot(gs[0, 0], projection='3d')
    ax.set_xticks([-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6])
    ax.set_yticks([-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6])
    ax.set_zticks([0.0,0.2,0.4,0.6])
    ax.axes.set_xlim3d(left=-0.6, right=0.6) 
    ax.axes.set_ylim3d(bottom=-0.6, top=0.6) 
    ax.axes.set_zlim3d(bottom=-0.0, top=0.6) 
    ax.axes.set_xlabel("X")
    ax.axes.set_ylabel("Y")
    ax.axes.set_zlabel("Z")
    ax.plot(points1[:,0], points1[:,1], points1[:,2])
    ax.scatter(points1[:,0], points1[:,1], points1[:,2])
    ax.set_title("Joints à l'origine (0)")

    ax = plt.subplot(gs[0, 1], projection='3d')
    ax.set_xticks([-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6])
    ax.set_yticks([-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6])
    ax.set_zticks([0.0,0.2,0.4,0.6])
    ax.axes.set_xlim3d(left=-0.6, right=0.6) 
    ax.axes.set_ylim3d(bottom=-0.6, top=0.6) 
    ax.axes.set_zlim3d(bottom=-0.0, top=0.6) 
    ax.axes.set_xlabel("X")
    ax.axes.set_ylabel("Y")
    ax.axes.set_zlabel("Z")
    ax.plot(points2[:,0], points2[:,1], points2[:,2])
    ax.scatter(points2[:,0], points2[:,1], points2[:,2])
    ax.set_title("Configuration: [45, 45, 45, 45, 20](deg), [0.03](m)")


def plot_drilling_bot_traj(robot, q):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    fig.suptitle("Trajectoire du bras")

    ax.set_xticks([-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6])
    ax.set_yticks([-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6])
    ax.set_zticks([0.0,0.2,0.4,0.6])

    ax.axes.set_xlabel("X")
    ax.axes.set_ylabel("Y")
    ax.axes.set_zlabel("Z")

    ax.axes.set_xlim3d(left=-0.8, right=0.8) 
    ax.axes.set_ylim3d(bottom=-0.8, top=0.8) 
    ax.axes.set_zlim3d(bottom=-0.0, top=0.6) 

    for i in range(len(q[0])):
        x = robot.forward_kinematic_lines(q[:,i])
        ax.plot(x[0][:,0], x[0][:,1], x[0][:,2], c='red')
        ax.plot(x[1][:,0], x[1][:,1], x[1][:,2], c='green')
        ax.scatter(x[1][:,0], x[1][:,1], x[1][:,2], c='blue')
        plt.pause(0.01)
    print("Done")


def plot_traj_cartesian(time, traj, speed, accel):
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure()
    fig.suptitle("Trajectoire de l'effecteur")

    ax = plt.subplot(gs[:, 0], projection='3d') # all rows, col 0
    ax.scatter(traj[0], traj[1], traj[2])
    ax.axes.set_xlabel("X")
    ax.axes.set_ylabel("Y")
    ax.axes.set_zlabel("Z")
    ax.set_title("Trajectoire cartésienne")

    ax = plt.subplot(gs[0, 1]) # row 0, col 1
    ax.plot(time, speed)
    ax.set_title("Vitesse de l'effecteur")
    ax.axes.set_xlabel("Temps (s)")
    ax.axes.set_ylabel("Vitesse (m/s)")

    ax = plt.subplot(gs[1, 1]) # row 1, col 1
    ax.plot(time, accel)
    ax.set_title("Accélération de l'effecteur")
    ax.axes.set_xlabel("Temps (s)")
    ax.axes.set_ylabel("Accel (m/s²)") 


def plot_traj_angular(time, q_traj, dq, ddq, tau):
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure()
    fig.suptitle("Trajectoire des joints")

    ax = plt.subplot(gs[0, 0]) # row 0, col 0
    ax.set_title("Configuration des joints")
    ax.axes.set_xlabel("Temps (s)")
    ax.axes.set_ylabel("Angle (rad)")
    plt.plot(time, q_traj[0])
    plt.plot(time, q_traj[1])
    plt.plot(time, q_traj[2])
    ax.legend(['q1', 'q2', 'q3'])

    ax = plt.subplot(gs[1, 0]) # row 1, col 0
    ax.set_title("Vitesse des joints")
    ax.axes.set_xlabel("Temps (s)")
    ax.axes.set_ylabel("Vitesse angulaire (rad/s)")
    plt.plot(time, dq[0])
    plt.plot(time, dq[1])
    plt.plot(time, dq[2])
    ax.legend(['dq1', 'dq2', 'dq3'])

    ax = plt.subplot(gs[0, 1]) # row 0, col 1
    ax.set_title("Accélération des joints")
    ax.axes.set_xlabel("Temps (s)")
    ax.axes.set_ylabel("Accel angulaire (rad/s²)")
    plt.plot(time, ddq[0])
    plt.plot(time, ddq[1])
    plt.plot(time, ddq[2])
    ax.legend(['ddq1', 'ddq2', 'ddq3'])

    ax = plt.subplot(gs[1, 1]) # row 1, col 1
    ax.set_title("Torque aux joints")
    ax.axes.set_xlabel("Temps (s)")
    ax.axes.set_ylabel("Torque (N/m)")
    plt.plot(time, tau[0])
    plt.plot(time, tau[1])
    plt.plot(time, tau[2])
    ax.legend(['tau1', 'tau2', 'tau3'])


def kuka_fwd_kin(q):
    # Robot DH Parameters
    d     = np.array([0.072, 0.075,        0,              0,     0,              0.217,   q[5]])
    theta = np.array([0,     np.pi + q[0], np.pi/2 - q[1], -q[2], np.pi/2 - q[3], -q[4],   0   ])
    r     = np.array([0,     0.033,        0.155,          0.136, 0,              0,       0   ])
    alpha = np.array([0,     np.pi/2,      0,              0,     np.pi/2,       -np.pi/2, 0   ])
    
    points = np.zeros((len(q) + 2, 3))
    points[0] = np.array([ 0, 0, 0 ])
    for i in range(len(q) + 1):
        # Compute transform matrix
        T = dhs2T( r[0:i+1], d[0:i+1], theta[0:i+1], alpha[0:i+1] )
        # End effector position
        points[i + 1] = np.array([ T[0][3], T[1][3], T[2][3] ])

    return points
 

def test_part1():
    # Test func dh2T
    r = 1
    d = 1 
    theta = np.pi
    alpha = np.pi
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    print(dh2T( r , d , theta, alpha ))

    # Test func dhs2T
    r = np.array([1, 2, 3, 4, 5])
    d = np.array([1, 2, 3, 4, 5])
    theta = np.array([1, 2, 3, 4, 5])
    alpha = np.array([1, 2, 3, 4, 5])
    print(dhs2T( r , d , theta, alpha ))

    # Test kuka forward kinematics
    q = [0, 0, 0, 0, 0, 0]
    points1 = kuka_fwd_kin(q)

    q = [45, 45, 45, 45, 20]
    x = [0.03]
    q = [ang*np.pi/180 for ang in q] + x
    points2 = kuka_fwd_kin(q)

    plot_kuka(points1, points2) 


def test_part2():
    # See commande_en_position_solution.py
    pass



def test_part3():
    # See commande_en_force.py
    pass



def test_part4():

    # Test goal2r
    p0 = np.array([0.5, 0.5, 0.4])
    p1 = np.array([0.55, 0.1, 0.5])
    # p1 = np.array([0.6, -0.3, 0.6])
    tf = 10
    l = 30
    time = np.linspace(0, tf, num=l, endpoint=True)

    traj, dr, ddr = goal2r(p0, p1, tf)
    
    speed = np.zeros(l)
    accel = np.zeros(l)

    for i in range(l):
        speed[i] = np.linalg.norm(dr[:,i])
        accel[i] = np.linalg.norm(ddr[:,i])
        accel_sign = np.sign(np.dot(p1-p0, ddr[:,i]))
        if not accel_sign == 1.0:
            accel[i] = -accel[i]
        
    # Test r2q
    robot = DrillingRobot()
    q_traj, dq, ddq = r2q( traj, dr, ddr , robot )

    # Test q2torque
    tau = q2torque(q_traj, dq, ddq , robot)
    
    # Plot results
    plot_traj_cartesian(time, traj, speed, accel)
    plot_traj_angular(time, q_traj, dq, ddq, tau)
    plot_drilling_bot_traj(robot, q_traj)



test_part1()
test_part4()
plt.show()