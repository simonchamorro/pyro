#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 28

@author: Simon Chamorro CHAS2436
------------------------------------

Devoir #5
"""

import numpy as np
import matplotlib.pyplot as plt


class TwolinkArm():
    def __init__(self, q=[0, 0], links=[3, 3]):
        self.links = links
        self.q = q
        self.Kq = np.array([[100000, 0], [0, 200000]])   
        self.Cq = np.linalg.inv(self.Kq)    
        self.compute_jacobian(self.q)
        
    def compute_jacobian(self, q):
        self.q = q
        l1 = self.links[0]
        l2 = self.links[1]
        c1 = np.cos(self.q[0])
        s1 = np.sin(self.q[0])
        c12 = np.cos(self.q[0] + self.q[1])
        s12 = np.sin(self.q[0] + self.q[1])

        # Jacobian
        self.J = np.array([[ (l1*c1 + l2*c12),  (l2*c12) ], \
                      [ -(l1*s1 + l2*s12), -(l2*s12) ]])
        self.J_t = np.transpose(self.J)

    def get_torques(self, force):
        torques = np.dot(self.J_t, force)
        return torques     

    def get_rigidiy_matrix(self):
        Kr = np.linalg.inv(np.dot(np.dot(self.J, self.Cq), self.J_t))
        return Kr

    def get_compliance_matrix(self):
        Cr = np.dot(np.dot(self.J, self.Cq), self.J_t)
        return Cr


# #5 Q1 ----- Forces vs Torques -----

arm = TwolinkArm()
f = np.array([0, -1])
q1 = range(-180, 180, 5)
q2 = range(-180, 180, 5)
x = []
y = []
torque_1 = []
torque_2 = []

for i in q1:
    for j in q2:
        arm.compute_jacobian(np.array([np.pi*i/180, np.pi*j/180]))
        t = arm.get_torques(f)
        x.append(i)
        y.append(j)
        torque_1.append(t[0])
        torque_2.append(t[1])

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x, y, torque_1, c=torque_1, cmap='Greens', s=1)
ax.set_title('Torque 1')

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x, y, torque_2, c=torque_2, cmap='Greens', s=1)
ax.set_title('Torque 2')


# #5 Q2 ----- Rigidity and Compliance -----
# a)
q = np.array([np.pi*30/180, np.pi*60/180])
arm.compute_jacobian(q)
print("-----------------------")
print("a) Rigidité cartésienne")
print(arm.get_rigidiy_matrix())

# b)
Cr = arm.get_compliance_matrix()
matrix = np.dot(np.transpose(Cr), Cr)
eig_values, eig_vectors = np.linalg.eig(matrix)
print("\n-----------------------")
print("b) Max/Min Compliance ")
print("vector: " + str(eig_vectors[0]) + ", value: " + str(eig_values[0]))
print("vector: " + str(eig_vectors[1]) + ", value: " + str(eig_values[1]))

# c)
min_compliance = []
max_compliance = []
for i in q1:
    for j in q2:
        arm.compute_jacobian(np.array([np.pi*i/180, np.pi*j/180]))
        Cr = arm.get_compliance_matrix()
        matrix = np.dot(np.transpose(Cr), Cr)
        eig_values, eig_vectors = np.linalg.eig(matrix)
        min_compliance.append(np.sqrt(min(eig_values)))
        max_compliance.append(np.sqrt(max(eig_values)))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x, y, min_compliance, c=min_compliance, s=1)
ax.set_title('Min Compliance')

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x, y, max_compliance, c=max_compliance, s=1)
ax.set_title('Max Compliance')

plt.show()