#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 28

@author: Simon Chamorro CHAS2436
------------------------------------

Devoir #7
Angular pos, speed and accel for given trajectory
Max speed 1m/s and max accel 1m/s**2
- p1 [5, 0]
- p2 [3, 0]
- p3 [0, 3]
- p4 [0, 5]

"""

import numpy as np
import matplotlib.pyplot as plt


class TwolinkArm():
    def __init__(self, q=[0.92729522, 1.57079633], links=[4, 3], make_plot=True):
        self.make_plot = make_plot
        self.links = links
        self.q = q
        self.k = 1
        self.forward_kinematics()

    def forward_kinematics(self):
        # Compute first joint position
        self.j1 = np.array([self.links[0]*np.sin(self.q[0]), \
                            self.links[0]*np.cos(self.q[0])])
        
        # Compute end effector position 
        self.end_effector = np.array([self.links[0]*np.sin(self.q[0]) \
                               + self.links[1]*np.sin(self.q[0] + self.q[1]), \
                                 self.links[0]*np.cos(self.q[0]) \
                               + self.links[1]*np.cos(self.q[0] + self.q[1])])

    def compute_jacobian(self):
        l1 = self.links[0]
        l2 = self.links[1]
        c1 = np.cos(self.q[0])
        s1 = np.sin(self.q[0])
        c12 = np.cos(self.q[0] + self.q[1])
        s12 = np.sin(self.q[0] + self.q[1])

        # Jacobian
        J = np.array([[ (l1*c1 + l2*c12),  (l2*c12) ], \
                      [ -(l1*s1 + l2*s12), -(l2*s12) ]])
        
        det_J = (l1*c1 + l2*c12)*-(l2*s12) - (l2*c12)*-(l1*s1 + l2*s12)
        
        # Detect singularities and inverse matrix
        assert not det_J == 0
        J_inv = np.linalg.inv(J)

        return J, J_inv

    def inverse_kinematics(self, dr):
        # Jacobian
        J, J_inv = self.compute_jacobian()

        # Find dq
        dq = np.dot(J_inv, dr)

        return dq

    def move_to(self, goal_pos, goal_speed, dt):
        # Jacobian
        J, J_inv = self.compute_jacobian()

        # Update robot position
        target = self.k*(goal_pos - self.end_effector)  + goal_speed
        q_prime = np.dot(J_inv, target)
        self.q = self.q + q_prime*dt
        self.forward_kinematics()
        
        # Update plot
        if self.make_plot:
            self.plot()
            self.print_state()

        return self.q, q_prime

    def plot(self):
        plt.clf()
        plt.axis([-7, 7, -7, 7])
        plt.plot([0, self.j1[0], self.end_effector[0]], \
                 [0, self.j1[1], self.end_effector[1]], color='blue')
        plt.scatter([0, self.j1[0]], [0, self.j1[1]], color='red')
        plt.scatter(self.end_effector[0], self.end_effector[1], color='green')
        plt.draw()
        plt.pause(1)

    def print_state(self):
        print("q: " + str(self.q))
        print("r: " + str(self.end_effector))



t5 = np.sqrt(18) - 2
diag = np.sqrt(0.5)
time = np.array([0, 1, 2, 3, 4, 4+t5, 5+t5, 6+t5, 7+t5, 8+t5])
r_pos = np.array([[5, 0], [4.5, 0], [3.5, 0], [3, 0], [3 - diag/2, diag/2],\
                 [diag/2, 3 - diag/2], [0, 3], [0, 3.5], [0, 4.5], [0, 5]])
r_speed = np.array([[0, 0], [-1, 0], [-1, 0], [0, 0], [-diag, diag],\
                   [-diag, diag], [0, 0], [0, 1], [0, 1], [0, 0]])

robot = TwolinkArm()

plt.figure()
plt.axis([-1, 7, -1, 7])
for t, p, s in zip(time, r_pos, r_speed):
    print(t)
    plt.scatter(p[0], p[1])
    plt.plot([p[0], p[0] + s[0]], [p[1], p[1] + s[1]])
    plt.pause(0.1)

q = []
dq = []
last_t = 0
plt.figure()
for t, p, s in zip(time, r_pos, r_speed):
    dt = t - last_t
    last_t = t
    qi, dqi = robot.move_to(p, s, dt)
plt.show()

