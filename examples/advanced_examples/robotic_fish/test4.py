import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spl

t_end = 10.
p0 = 0.
# pump_max_pressure = 3.5e4
pump_max_pressure = 2.e5

pump_inductance = 1.e6
# pump_inductance = 3.e5
# pump_inductance = 3.e4
pump_resistance = 1.e6
actuator_capacitance = 1.e-6
actuator_resistance = 1.e6

# pump_inductance = 1.
# pump_resistance = 1.
# actuator_capacitance = 1.
# actuator_resistance = 1.

time = np.linspace(0, t_end, 1000)
current_rl = pump_max_pressure/pump_resistance*(1-np.exp(-time/(pump_inductance/pump_resistance)))

plt.plot(time, current_rl)
plt.show()

pressure_rc = pump_max_pressure*(1-np.exp(-time/(actuator_resistance*actuator_capacitance)))
plt.plot(time, pressure_rc, label='RC Prediction')

L = pump_inductance
R = pump_resistance + actuator_resistance
C = actuator_capacitance

A = np.array([[0, 1], 
              [-1/(L*C), -R/L]])
B = np.array([[0], [1]])

x0 = np.array([[0], [0]])
pressure = np.zeros((len(time), 2))
for i, t in enumerate(time):
    # pressure[i,:] = (spl.expm(A*t).dot(x0) + np.linalg.inv(A).dot(spl.expm(A*t)-np.eye(2)).dot(B).dot(pump_max_pressure)).reshape((2,))
    pressure[i,:] = (spl.expm(A*t).dot(x0) + np.linalg.inv(A).dot(spl.expm(A*t)-np.eye(2)).dot(B).dot(pump_max_pressure)).reshape((2,))

print(np.linalg.eig(A))

plt.plot(time, pressure[:,0], label='Pressure')
plt.plot(time, pressure[:,1], label='Pressure Derivative')
plt.legend()
plt.show()