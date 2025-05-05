import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dm4bem

##Room
l = 6               # m length of the rectangular room
L=3.06              # m width of the rectangular room
H=3                 # m heigh of the room
w=0.28              # m Thickness of the wall 
S_floor=l*L         # m² Surface of the floor
S_w1=L*H            # m² Surface of the opposite wall
S_w2=l*H            # m² Surface of the latteral walls


##Glass
l_g=2.5             # m lenght of the glass
H_g=3               # m Heigh of the glass
S_g=l_g*H_g         # m² Surface of the glass


##Materials
air = {'Density': 1.2,                      # kg/m³
       'Specific heat': 1000}               # J/(kg·K)
pd.DataFrame(air, index=['Air'])

concrete_floor = {'Conductivity': 1.400,    # W/(m·K)
            'Density': 2300.0,              # kg/m³
            'Specific heat': 880,           # J/(kg⋅K)
            'Width': 0.3,                  # m
            'Surface': S_floor}             # m²

insulation_floor = {'Conductivity': 0.027,  # W/(m·K)
              'Density': 55.0,              # kg/m³
              'Specific heat': 1210,        # J/(kg⋅K)
              'Width': 0,                   # m
              'Surface': S_floor}           # m²

concrete_lat= {'Conductivity': 1.400,       # W/(m·K)
            'Density': 2300.0,              # kg/m³
            'Specific heat': 880,           # J/(kg⋅K)
            'Width': 0.2,                   # m
            'Surface': S_w2*2}              # m²

insulation_lat = {'Conductivity': 0.027,    # W/(m·K)
              'Density': 55.0,              # kg/m³
              'Specific heat': 1210,        # J/(kg⋅K)
              'Width': 0.08,                # m
              'Surface': S_w2*2}            # m²

concrete_op = {'Conductivity': 1.400,       # W/(m·K)
            'Density': 2300.0,              # kg/m³
            'Specific heat': 880,           # J/(kg⋅K)
            'Width': 0.15,                   # m
            'Surface': S_w1}                # m²

insulation_op = {'Conductivity': 0.027,     # W/(m·K)
              'Density': 55.0,              # kg/m³
              'Specific heat': 1210,        # J/(kg⋅K)
              'Width': 0.13,                # m
              'Surface': S_w1}              # m²


glass = {'Conductivity': 1.4,               # W/(m·K)
         'Density': 2500,                   # kg/m³
         'Specific heat': 1210,             # J/(kg⋅K)
         'Width': 0.04,                     # m
         'Surface': S_g}                   # m²

wall_floor = pd.DataFrame.from_dict({'wall': 'floor', 'Layer_out': concrete_floor,
                               'Layer_in': insulation_floor,},
                              orient='index')
wall_floor

wall_lat = pd.DataFrame.from_dict({'wall': 'lat','Layer_out': concrete_lat,
                               'Layer_in': insulation_lat,},
                              orient='index')
wall_lat

wall_op = pd.DataFrame.from_dict({'wall': 'op','Layer_out': concrete_op,
                               'Layer_in': insulation_op,},
                              orient='index')
wall_op

wall_glass=pd.DataFrame.from_dict({'wall': 'glass','Layer_out':glass},orient = 'index')
wall_glass





# radiative properties
ε_wLW = 0.85    # long wave emmisivity: wall surface (concrete)
ε_gLW = 0.90    # long wave emmisivity: glass pyrex
α_wSW = 0.25    # short wave absortivity: white smooth surface
α_gSW = 0.38    # short wave absortivity: reflective blue glass
τ_gSW = 0.30    # short wave transmitance: reflective blue glass
σ = 5.67e-8     # W/(m²⋅K⁴) Stefan-Bolzmann constant

h = pd.DataFrame([{'in': 8., 'out': 25}], index=['h'])  # W/(m²⋅K)
h


# conduction
G_cd= ((wall_floor['Conductivity']['Layer_in'] / wall_floor['Width']['Layer_in']) + 
       wall_floor['Conductivity']['Layer_out'] / wall_floor['Width']['Layer_out']* wall_floor['Surface'])

G_cd+= (wall_glass['Conductivity'] / wall_glass['Width'] +
        wall_glass['Conductivity'] / wall_glass['Width']) * wall_glass['Surface']

G_cd+= (wall_lat['Conductivity'] / wall_lat['Width']+
        wall_lat['Conductivity'] / wall_lat['Width']) * wall_lat['Surface']

G_cd+= (wall_op['Conductivity'] / wall_op['Width']+
        wall_op['Conductivity'] / wall_op['Width']) * wall_op['Surface']

pd.DataFrame(G_cd, columns=['Conductance'])



# convection
Gw_floor = h * wall_floor['Surface']
Gw_glass = h * wall_glass['Surface']
Gw_lat =  h * wall_lat['Surface']
Gw_op =  h * wall_op['Surface']


# view factor wall-glass
Fwg = glass['Surface'] / (concrete_floor['Surface']+concrete_lat['Surface']+concrete_op['Surface'])

T_int = 273.15 + np.array([0, 40])
coeff = np.round((4 * σ * T_int**3), 1)
print(f'For 0°C < (T/K - 273.15)°C < 40°C, 4σT³/[W/(m²·K)] ∈ {coeff}')

T_int = 273.15 + np.array([10, 30])
coeff = np.round((4 * σ * T_int**3), 1)
print(f'For 10°C < (T/K - 273.15)°C < 30°C, 4σT³/[W/(m²·K)] ∈ {coeff}')

T_int = 273.15 + 20
coeff = np.round((4 * σ * T_int**3), 1)
print(f'For (T/K - 273.15)°C = 20°C, 4σT³ = {4 * σ * T_int**3:.1f} W/(m²·K)')


# long wave radiation
Tm = 20 + 273   # K, mean temp for radiative exchange

GLW1 = 4 * σ * Tm**3 * ε_wLW / (1 - ε_wLW) * (wall_floor['Surface']['Layer_in']+
                                              wall_lat['Surface']['Layer_in']+
                                              wall_op['Surface']['Layer_in'])

GLW12 = 4 * σ * Tm**3 * Fwg * (wall_floor['Surface']['Layer_in']+
                                              wall_lat['Surface']['Layer_in']+
                                              wall_op['Surface']['Layer_in'])

GLW2 = 4 * σ * Tm**3 * ε_gLW / (1 - ε_gLW) * wall_glass['Surface'][glass]

GLW = 1 / (1 / GLW1 + 1 / GLW12 + 1 / GLW2)

# ventilation flow rate
Va = l**3                   # m³, volume of air
ACH = 1                     # 1/h, air changes per hour
Va_dot = ACH / 3600 * Va    # m³/s, air infiltration

# ventilation & advection
Gv = air['Density'] * air['Specific heat'] * Va_dot


# P-controler gain
# Kp = 1e4            # almost perfect controller Kp -> ∞
# Kp = 1e-3           # no controller Kp -> 0
Kp = 0



# glass: convection outdoor & conduction
Ggs = float(1 / (1 / Gg.loc['h', 'out'] + 1 / (2 * G_cd['Glass'])))

C = wall['Density'] * wall['Specific heat'] * wall['Surface'] * wall['Width']
pd.DataFrame(C, columns=['Capacity'])



C['Air'] = air['Density'] * air['Specific heat'] * Va
pd.DataFrame(C, columns=['Capacity'])

# temperature nodes
θ = ['θ0', 'θ1', 'θ2', 'θ3', 'θ4', 'θ5', 'θ6', 'θ7']

# flow-rate branches
q = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11']

# temperature nodes
nθ = 8      # number of temperature nodes
θ = [f'θ{i}' for i in range(8)]

# flow-rate branches
nq = 12     # number of flow branches
q = [f'q{i}' for i in range(12)]


A = np.zeros([12, 8])       # n° of branches X n° of nodes
A[0, 0] = 1                 # branch 0: -> node 0
A[1, 0], A[1, 1] = -1, 1    # branch 1: node 0 -> node 1
A[2, 1], A[2, 2] = -1, 1    # branch 2: node 1 -> node 2
A[3, 2], A[3, 3] = -1, 1    # branch 3: node 2 -> node 3
A[4, 3], A[4, 4] = -1, 1    # branch 4: node 3 -> node 4
A[5, 4], A[5, 5] = -1, 1    # branch 5: node 4 -> node 5
A[6, 4], A[6, 6] = -1, 1    # branch 6: node 4 -> node 6
A[7, 5], A[7, 6] = -1, 1    # branch 7: node 5 -> node 6
A[8, 7] = 1                 # branch 8: -> node 7
A[9, 5], A[9, 7] = 1, -1    # branch 9: node 5 -> node 7
A[10, 6] = 1                # branch 10: -> node 6
A[11, 6] = 1                # branch 11: -> node 6

pd.DataFrame(A, index=q, columns=θ)


G = np.array(np.hstack(
    [Gw['out'],
     2 * G_cd['Layer_out'], 2 * G_cd['Layer_out'],
     2 * G_cd['Layer_in'], 2 * G_cd['Layer_in'],
     GLW,
     Gw['in'],
     Gg['in'],
     Ggs,
     2 * G_cd['Glass'],
     Gv,
     Kp]))

# np.set_printoptions(precision=3, threshold=16, suppress=True)
# pd.set_option("display.precision", 1)
pd.DataFrame(G, index=q)


neglect_air_glass = False

if neglect_air_glass:
    C = np.array([0, C['Layer_out'], 0, C['Layer_in'], 0, 0,
                  0, 0])
else:
    C = np.array([0, C['Layer_out'], 0, C['Layer_in'], 0, 0,
                  C['Air'], C['Glass']])

# pd.set_option("display.precision", 3)
pd.DataFrame(C, index=θ)


b = pd.Series(['To', 0, 0, 0, 0, 0, 0, 0, 'To', 0, 'To', 'Ti_sp'],
              index=q)


f = pd.Series(['Φo', 0, 0, 0, 'Φi', 0, 'Qa', 'Φa'],
              index=θ)

y = np.zeros(8)         # nodes
y[[6]] = 1              # nodes (temperatures) of interest
pd.DataFrame(y, index=θ)


# thermal circuit
A = pd.DataFrame(A, index=q, columns=θ)
G = pd.Series(G, index=q)
C = pd.Series(C, index=θ)
b = pd.Series(b, index=q)
f = pd.Series(f, index=θ)
y = pd.Series(y, index=θ)

TC = {"A": A,
      "G": G,
      "C": C,
      "b": b,
      "f": f,
      "y": y}