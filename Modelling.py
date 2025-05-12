import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dm4bem
from IPython.display import display
from matrice_A import Matrice


##Room
L = 6               # m length of the rectangular room
l = 3.06              # m width of the rectangular room
H = 3                 # m heigh of the room
w = 0.28              # m Thickness of the wall 
S_floor = l*L         # m² Surface of the floor
S_w1 = l*H            # m² Surface of the opposite wall
S_w2 = L*H            # m² Surface of the latteral walls
S_tot_w2 = 2*S_w2     # m² Total surface of the latteral walls


##Glass
l_g = 2.5             # m lenght of the glass
H_g = 3               # m Heigh of the glass
S_g = l_g*H_g         # m² Surface of the glass

A=Matrice()

"""A partir d'ici je ne suis pas certain de ce que j'ai fait en particulier sur la définition des différentes couches"""

##Materials
air = {'Density': 1.2,                      # kg/m³
       'Specific heat': 1000}               # J/(kg·K)
pd.DataFrame(air, index=['Air'])

concrete_floor = {'Conductivity': 1.400,    # W/(m·K)
            'Density': 2300.0,              # kg/m³
            'Specific heat': 880,           # J/(kg⋅K)
            'Width': 0.3,                  # m
            'Surface': S_floor}             # m²

insulation_floor = {'Conductivity': 0.036,  # W/(m·K)
              'Density': 55.0,              # kg/m³
              'Specific heat': 1210,        # J/(kg⋅K)
              'Width': 0.001,               # m
              'Surface': S_floor}           # m²

concrete_lat= {'Conductivity': 1.400,       # W/(m·K)
            'Density': 2300.0,              # kg/m³
            'Specific heat': 880,           # J/(kg⋅K)
            'Width': 0.2,                   # m
            'Surface': S_w2*2}              # m²

insulation_lat = {'Conductivity': 0.036,    # W/(m·K)
              'Density': 55.0,              # kg/m³
              'Specific heat': 1210,        # J/(kg⋅K)
              'Width': 0.08,                # m
              'Surface': S_w2*2}            # m²

concrete_op = {'Conductivity': 1.400,       # W/(m·K)
            'Density': 2300.0,              # kg/m³
            'Specific heat': 880,           # J/(kg⋅K)
            'Width': 0.15,                   # m
            'Surface': S_w1}                # m²

insulation_op = {'Conductivity': 0.036,     # W/(m·K)
              'Density': 55.0,              # kg/m³
              'Specific heat': 1210,        # J/(kg⋅K)
              'Width': 0.13,                # m
              'Surface': S_w1}              # m²


glass = {'Conductivity': 1.4,               # W/(m·K)
         'Density': 2500,                   # kg/m³
         'Specific heat': 1210,             # J/(kg⋅K)
         'Width': 0.04,                     # m
         'Surface': S_g}                   # m²


wall = pd.DataFrame.from_dict({'Layer_1': concrete_floor,
                               'Layer_2': insulation_floor,
                               'Layer_3':concrete_lat,
                               'Layer_4':insulation_lat,
                               'Layer_5':concrete_op,
                               'Layer_6':insulation_op,
                               'Layer_7':glass},
                              orient='index')
wall





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
G_cd = wall['Conductivity'] / wall['Width'] * wall['Surface']
G_cd['Layer_2']=0
pd.DataFrame(G_cd, columns=['Conductance'])


# convection
Gw1 = h * wall['Surface']['Layer_1']     # wall
Gw2 = h * wall['Surface']['Layer_3']
Gw3 = h * wall['Surface']['Layer_5']

Gg = h * wall['Surface']['Layer_7']     # glass


# view factor wall-glass
Fwg = glass['Surface'] / (concrete_floor['Surface']+concrete_lat['Surface']+concrete_op['Surface'])

T_int = 273.15 + np.array([0, 40])
coeff = np.round((4 * σ * T_int**3), 1)
#print(f'For 0°C < (T/K - 273.15)°C < 40°C, 4σT³/[W/(m²·K)] ∈ {coeff}')

T_int = 273.15 + np.array([10, 30])
coeff = np.round((4 * σ * T_int**3), 1)
#print(f'For 10°C < (T/K - 273.15)°C < 30°C, 4σT³/[W/(m²·K)] ∈ {coeff}')

T_int = 273.15 + 20
coeff = np.round((4 * σ * T_int**3), 1)
#print(f'For (T/K - 273.15)°C = 20°C, 4σT³ = {4 * σ * T_int**3:.1f} W/(m²·K)')


# long wave radiation
Tm = 20 + 273   # K, mean temp for radiative exchange

GLW1_1 = 4 * σ * Tm**3 * ε_wLW / (1 - ε_wLW) * (wall['Surface']['Layer_2'])
GLW1_2 = 4 * σ * Tm**3 * ε_wLW / (1 - ε_wLW) * (wall['Surface']['Layer_4'])
GLW1_3 = 4 * σ * Tm**3 * ε_wLW / (1 - ε_wLW) * (wall['Surface']['Layer_6'])

GLW12_1 = 4 * σ * Tm**3 * Fwg * (wall['Surface']['Layer_1'])
GLW12_2 = 4 * σ * Tm**3 * Fwg * (wall['Surface']['Layer_3'])
GLW12_3 = 4 * σ * Tm**3 * Fwg * (wall['Surface']['Layer_5'])

GLW2 = 4 * σ * Tm**3 * ε_gLW / (1 - ε_gLW) * wall['Surface']['Layer_7']

GLW_1 = 1 / (1 / GLW1_1 + 1 / GLW12_1 + 1 / GLW2)
GLW_2 = 1 / (1 / GLW1_2 + 1 / GLW12_2 + 1 / GLW2)
GLW_3 = 1 / (1 / GLW1_3 + 1 / GLW12_3 + 1 / GLW2)

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
Ggs = float(1 / (1 / Gg.loc['h', 'out'] + 1 / (2 * G_cd['Layer_7'])))

C = wall['Density'] * wall['Specific heat'] * wall['Surface'] * wall['Width']
pd.DataFrame(C, columns=['Capacity'])



C['Air'] = air['Density'] * air['Specific heat'] * Va
pd.DataFrame(C, columns=['Capacity'])

# temperature nodes
θ = ['θ0', 'θ1', 'θ2', 'θ3', 'θ4', 'θ5', 'θ6', 'θ7','θ8','θ9','θ10','θ11','θ12','θ13','θ14','θ15','θ16','θ17']

# flow-rate branches
q = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11','q12','q13','q14','q15','q16','q17','q18','q19','q20','q21','q22','q23','q24','q25']
# temperature nodes
nθ =  18     # number of temperature nodes
θ = [f'θ{i}' for i in range(nθ)]

# flow-rate branches
nq = 26     # number of flow branches
q = [f'q{i}' for i in range(nq)]


A = Matrice()
pd.DataFrame(A, index=q, columns=θ)



G = np.array(np.hstack(
    [Gw1['out'],
     2 * G_cd['Layer_1'], 2 * G_cd['Layer_1'],
     2 * G_cd['Layer_2'], 2 * G_cd['Layer_2'],
     GLW_1,    
     Gw1['in'], #q6
     Gg['in'],
     Ggs,                   
     2 * G_cd['Layer_7'],   #q9
     Gv,
     Kp,
     Gw2['out'], #q12
     2 * G_cd['Layer_3'], 2 * G_cd['Layer_3'],
     2 * G_cd['Layer_4'], 2 * G_cd['Layer_4'],
     Gw3['out'], #q17
     2 * G_cd['Layer_5'], 2 * G_cd['Layer_5'],
     2 * G_cd['Layer_6'], 2 * G_cd['Layer_6'],
     GLW_2, #q22
     GLW_3,
     Gw2['in'],
     Gw3['in'],
     ]))

# np.set_printoptions(precision=3, threshold=16, suppress=True)
# pd.set_option("display.precision", 1)
pd.DataFrame(G, index=q)


neglect_air_glass = False

if neglect_air_glass:
    C = np.array([0, C['Layer_1'], 0, C['Layer_2'], 0, 0,
                  0, 0, 0, C['Layer_3'], 0, C['Layer_4'], 0, 0, C['Layer_5'], 0, C['Layer_6'], 0])
else:
    C = np.array([0, C['Layer_1'], 0, C['Layer_2'], 0, C['Air'],
                  C['Layer_7'], 0, 0, C['Layer_3'], 0, C['Layer_4'], 0, 0, C['Layer_5'], 0, C['Layer_6'], 0])



# pd.set_option("display.precision", 3)
pd.DataFrame(C, index=θ)


b = pd.Series(['To', 0, 0, 0, 0, 0, 0, 0, 'To', 0, 'To', 'Ti_sp', 'To', 0, 0, 0, 0, 'To', 0, 0, 0, 0, 0, 0, 0, 0 ],
              index=q)


f = pd.Series(['Φo', 0, 0, 0, 'Φi', 0, 'Qa', 'Φa', 'Φo', 0, 0, 0, 'Φi', 'Φo', 0, 0, 0, 'Φi'],
              index=θ)

y = np.zeros(nθ)         # nodes
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


#TC['G']['q11'] = 1e3  # Kp -> ∞, almost perfect controller
TC['G']['q11'] = 0      # Kp -> 0, no controller (free-floating)


[As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)
us
