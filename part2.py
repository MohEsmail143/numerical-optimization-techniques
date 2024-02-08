#!/usr/bin/env python
# coding: utf-8

# # Optimization Mini-Project

# Importing Libraries

# In[1]:


import numpy as np
import sympy as sp


# ## Part II: Numerical Constrained Optimization Techniques
# 
# You should write a program that solves constrained optimization problems using the penalty function method studied in class with gradient descent method for unconstrained minimization. Clearly, describe the parameters settings. Test your program and provide sample runs in the report accompanying your implementation.

# ### Exterior Penalty Method

# In[2]:


def exterior_penalty(problem, x0, r0, descent_iterations=1000, num_iterations=3, tolerance=0.25, c=10, lambda_=0.01):
    r = r0
    i = 0
    f = problem['f']
    h = 0
    for h_i in problem['h']:
        h += sp.Pow(h_i, 2)
    g = 0
    for g_i in problem['g']:
        g += sp.Pow(sp.Max(0, g_i), 2)
    j = 0
    while j < num_iterations:
        i = 0
        violated_constraint = False
        x_val, y_val = x0
        phi = f + r * h + r * g
        del_phi_x = sp.diff(phi, x)
        del_phi_y = sp.diff(phi, y)
        while i < descent_iterations:
            violated_constraint = False
            del_phi_x_k = del_phi_x.subs([(x, x_val), (y, y_val)])
            del_phi_y_k = del_phi_y.subs([(x, x_val), (y, y_val)])
            x_val -= lambda_ * del_phi_x_k
            y_val -= lambda_ * del_phi_y_k
            for h_i in problem['h']:
                if not (-tolerance <= h_i.subs([(x, x_val), (y, y_val)]) <= tolerance):
                    violated_constraint = True
                    break
            for g_i in problem['g']:
                if g_i.subs([(x, x_val), (y, y_val)]) > tolerance:
                    violated_constraint = True
                    break         
            if violated_constraint == False:
                break
            i += 1
        if violated_constraint:
            r *= c
        else:
            return x_val, y_val
        j += 1
    return round(x_val, 2), round(y_val, 2)


# ### Examples of Objective Functions
# 
# h: equality equations
# 
# g: inequality equations

# In[3]:


def state_problem(obj_func, g_constraints=None, h_constraints=None):
    problem = dict()
    problem['g'] = []
    problem['h'] = []
    problem['f'] = obj_func
    if g_constraints is not None:
        for i in range(len(g_constraints)):
            problem['g'].append(g_constraints[i])
    if h_constraints is not None:
        for i in range(len(h_constraints)):    
            problem['h'].append(h_constraints[i])        
    return problem


# In[4]:


x, y = sp.symbols('x y', real=True)


# **Example 1:**
# 
# Minimize $f(x, y) = 3x^ 2 + 4y^2, $ subject to $x + 2y = 8, $ for 3 iterations and setting the penalty parameter r = 1 and the scaling parameter c = 10

# In[5]:


ex1 = state_problem(3*x**2 + 4*y**2, 'min', h_constraints=[x + 2*y - 8])
ex1


# **Example 2:**
# 
# 
# Minimize $f(x, y) = 2x + (y-3)^2, $ subject to $x >= 3, y >= 3, $ for 2 iterations and setting the penalty parameter r = 1 and the scaling parameter c = 10

# In[6]:


ex2 = state_problem(2*x + (y-3)**2, 'min', g_constraints=[3 - x, 3 - y])
ex2


# **Example 3:**
# 
# Minimize $f(x,y) = (1/3) * (x+1)^3 + y, $ subject to $x >= 1, y >= 0, $ for 2 iterations and setting the penalty parameter r = 1 and the scaling parameter c = 10

# In[7]:


ex3 = state_problem((1/3) * (x+1)**3 + y, 'min', g_constraints=[1-x, -y])
ex3


# ### Solving the Examples

# Analytic Solution: $(2, 3)$

# In[11]:


res = exterior_penalty(ex1, (1,1), 1, tolerance=0.05, lambda_=0.001)
print(res)


# Analytic Solution: $(3, 3)$

# In[12]:


res = exterior_penalty(ex2, (2,2), 1, tolerance=0.01, lambda_=0.001)
print(res)


# Analytic Solution= $(1, 0)$

# In[13]:


res = exterior_penalty(ex3, (0,0), 1, tolerance=0.01, lambda_=0.001)
print(res)

