#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 28

@author: Simon Chamorro CHAS2436
------------------------------------

Devoir #3 d)
Follow a circular trajectory following x = 1.5*np.cos(0.5*(t**2)) 
									   y = 1.5*np.sin(0.5*(t**2))

"""

import numpy as np
import matplotlib.pyplot as plt


class TwolinkArm():
    def __init__(self, q=[0.25268, 2.6362], links=[3, 3], make_plot=True):
        self.make_plot = make_plot
        self.links = links
        self.q = q
        self.k = 1
        self.forward_kinematics()
        if self.make_plot:
            plt.figure()
            self.plot()
            self.print_state()

    def forward_kinematics(self):
    	# Compute first joint position
        self.j1 = np.array([self.links[0]*np.sin(self.q[0]), \
                            self.links[0]*np.cos(self.q[0])])
        
        # Compute end effector position 
        self.end_effector = np.array([self.links[0]*np.sin(self.q[0]) \
                               + self.links[1]*np.sin(self.q[0] + self.q[1]), \
                                 self.links[0]*np.cos(self.q[0]) \
                               + self.links[1]*np.cos(self.q[0] + self.q[1])])

    def move_to(self, goal_pos, goal_speed, dt):
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

        # Update robot position
        target = self.k*(goal_pos - self.end_effector)  + goal_speed
        q_prime = np.dot(J_inv, target)
        self.q = self.q + q_prime*dt
        self.forward_kinematics()
        
        # Update plot
        if self.make_plot:
            self.plot()
            self.print_state()

    def plot(self):
        plt.clf()
        plt.axis([-7, 7, -7, 7])
        plt.plot([0, self.j1[0], self.end_effector[0]], \
                 [0, self.j1[1], self.end_effector[1]], color='blue')
        plt.scatter([0, self.j1[0]], [0, self.j1[1]], color='red')
        plt.scatter(self.end_effector[0], self.end_effector[1], color='green')
        plt.draw()
        plt.pause(0.03)

    def print_state(self):
        print("q: " + str(self.q))
        print("r: " + str(self.end_effector))

    
arm = TwolinkArm()
dt = 0.05
end_time = 10.0
time = np.linspace(0, end_time, endpoint=False, num=int(end_time/dt))
for t in time:
    print("t: " + str(t))
    goal_pos = np.array([1.5*np.cos(0.5*(t**2)), 1.5*np.sin(0.5*(t**2))])
    goal_speed = np.array([-1.5*t*np.sin(t**2/2), 1.5*t*np.cos(t**2/2)])
    arm.move_to(goal_pos, goal_speed, dt)
