from sympy import *

def computeE(lr, u_start, v_start, num_update):
    E = symbols('E', cls=Function)
    u, v = symbols('u v', commutative=True)
    E = exp(u) + exp(2*v) + exp(u*v) + u**2 - 2*u*v + 2*v**2 - 3*u - 2*v
    u_grad, v_grad = diff(E, u), diff(E, v)
    u_grad_value = u_grad.subs([(u, u_start), (v, v_start)])
    v_grad_value = v_grad.subs([(u, u_start), (v, v_start)])

    u_value, v_value = u_start, v_start
    for n in range(num_update):
        new_u = u_value - lr*u_grad.subs([(u, u_value), (v, v_value)])
        new_v = v_value - lr*v_grad.subs([(u, u_value), (v, v_value)])
        u_value, v_value = new_u, new_v
    
    E_value = E.subs([(u, u_value), (v, v_value)])
    return u_grad_value, v_grad_value, E_value

lr = 0.01
u_grad_value, v_grad_value, _ = computeE(lr, 0, 0, 1)
print("Answer of question 6: ({}, {})".format(u_grad_value , v_grad_value))

_, _, E_value = computeE(lr, 0, 0, 5)
print("Answer of question 7: E(v_5, u_5) = ", E_value)


   