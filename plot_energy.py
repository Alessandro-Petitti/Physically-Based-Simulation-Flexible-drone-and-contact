#!/usr/bin/env python3
"""
Script per calcolare e plottare l'energia totale del sistema multi-body nel tempo.

Sistema: 5 corpi rigidi (base + 4 bracci) con giunti sferici 3DOF.
Nei giunti: molla torsionale + smorzatore: τ_i = -k*e_i - b*ω_rel,i

L'energia totale è:
    E_tot = T_base + Σ_i T_i + V_grav + Σ_i V_spring,i

dove:
- T_base = 0.5*m_B*||W_v_B||² + 0.5*B_ω_B^T*I_B*B_ω_B  (energia cinetica base)
- T_i = 0.5*m_i*||W_v_P_i||² + 0.5*P_ω_P_i^T*I_P*P_ω_P_i  (energia cinetica braccio i)
- V_grav = Σ_j m_j*g*z_j  (potenziale gravitazionale di ogni corpo)
- V_spring,i = 0.5*k*||e_i||²  (potenziale elastico molla torsionale)
  con e_i = Log(R_BH0^T * R_BH) ∈ R³ (rotation vector dell'errore in SO(3))

⚠️ Lo smorzatore (-b*ω_rel) NON ha energia associata, la dissipa.
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml


def load_parameters(param_file='model/drone_parameters.yaml'):
    """Carica i parametri del drone dal file YAML."""
    with open(param_file, 'r') as f:
        params = yaml.safe_load(f)
    return params


def load_frame(frame_path):
    """Carica un frame dal file JSON."""
    with open(frame_path, 'r') as f:
        data = json.load(f)
    return data


def quat_to_rotmat(q_wxyz):
    """
    Converte un quaternione [w, x, y, z] in matrice di rotazione 3x3.
    """
    w, x, y, z = q_wxyz
    # Normalizza
    n = np.sqrt(w*w + x*x + y*y + z*z)
    if n > 1e-12:
        w, x, y, z = w/n, x/n, y/n, z/n
    
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])
    return R


def skew(v):
    """Matrice antisimmetrica (skew-symmetric) di un vettore 3D."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def logSO3(R):
    """
    Logaritmo di una matrice di rotazione: Log: SO(3) -> R³ (rotation vector).
    Restituisce θ*u dove θ è l'angolo e u è l'asse di rotazione.
    """
    tr = np.trace(R)
    cos_th = 0.5 * (tr - 1.0)
    cos_th = np.clip(cos_th, -1.0, 1.0)
    th = np.arccos(cos_th)
    
    # Caso angolo piccolo
    if th < 1e-6:
        # Approssimazione: Log(R) ≈ 0.5 * vee(R - R^T)
        S = R - R.T
        return 0.5 * np.array([S[2,1], S[0,2], S[1,0]])
    
    # Caso vicino a π
    if th > np.pi - 1e-3:
        # Usa l'estrazione dalla diagonale
        rx = np.sqrt(max(0.0, (R[0,0] + 1.0) * 0.5))
        ry = np.sqrt(max(0.0, (R[1,1] + 1.0) * 0.5))
        rz = np.sqrt(max(0.0, (R[2,2] + 1.0) * 0.5))
        rx = np.copysign(rx, R[2,1] - R[1,2])
        ry = np.copysign(ry, R[0,2] - R[2,0])
        rz = np.copysign(rz, R[1,0] - R[0,1])
        u = np.array([rx, ry, rz])
        if np.linalg.norm(u) > 1e-12:
            u = u / np.linalg.norm(u)
        return th * u
    
    # Caso generale
    s_th = np.sin(th)
    S = R - R.T
    veeS = np.array([S[2,1], S[0,2], S[1,0]])
    return (th / (2.0 * s_th)) * veeS


def parse_transform(t_data):
    """
    Converte [x, y, z, qw, qx, qy, qz] in matrice 4x4 omogenea.
    """
    x, y, z = t_data[0], t_data[1], t_data[2]
    qw, qx, qy, qz = t_data[3], t_data[4], t_data[5], t_data[6]
    R = quat_to_rotmat([qw, qx, qy, qz])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def calculate_energy_from_frame(frame_data, params, T_BH_list, T_HP_list):
    """
    Calcola l'energia totale del sistema a partire dai dati di un frame.
    
    Sistema: 5 corpi rigidi (base + 4 bracci) con giunti sferici.
    
    Args:
        frame_data: Dizionario con i dati del frame (state, T_WB, T_WP, etc.)
        params: Parametri del drone
        T_BH_list: Lista delle trasformazioni T_BH (base -> hinge) dal YAML
        T_HP_list: Lista delle trasformazioni T_HP (hinge -> prop) dal YAML
    
    Returns:
        Dizionario con tutte le componenti di energia
    """
    state = np.array(frame_data['state'])
    sim_time = frame_data.get('simTime', 0.0)
    
    # ========================================
    # Estrai componenti dallo stato
    # ========================================
    # Stato: [p_WB(3), q_base(4), v_WB(3), ω_B(3), q_arm0(4), q_arm1(4), q_arm2(4), q_arm3(4), ω_rel0(3), ω_rel1(3), ω_rel2(3), ω_rel3(3)]
    
    p_WB = state[0:3]           # Posizione base (world frame)
    q_base = state[3:7]         # Quaternione base [w, x, y, z]
    v_WB = state[7:10]          # Velocità lineare base (world frame)
    B_omega_B = state[10:13]    # Velocità angolare base (body frame)
    
    # Quaternioni dei bracci (R_H0H: rotazione relativa dal frame di riposo H0 al corrente H)
    q_arms = [state[13+4*i:17+4*i] for i in range(4)]
    
    # Velocità angolari relative dei bracci (nel frame del braccio P)
    P_omega_rel = [state[29+3*i:32+3*i] for i in range(4)]
    
    # ========================================
    # Parametri fisici
    # ========================================
    m_base = params['mass']['base']
    m_arm = (params['mass']['total'] - params['mass']['base']) / 4.0
    g = 9.8066
    k_joint = params['morphing_joint']['k_joint']  # Costante molla torsionale
    
    # Matrici di inerzia (diagonali)
    I_base = np.diag([
        params['inertia']['B_Ixx_B'],
        params['inertia']['B_Iyy_B'],
        params['inertia']['B_Izz_B']
    ])
    
    I_arm = np.diag([
        params['inertia']['P_Ixx_P'],
        params['inertia']['P_Iyy_P'],
        params['inertia']['P_Izz_P']
    ])
    
    # ========================================
    # Matrici di rotazione
    # ========================================
    R_WB = quat_to_rotmat(q_base)
    W_omega_B = R_WB @ B_omega_B  # Velocità angolare base in world frame
    
    # ========================================
    # 1. ENERGIA CINETICA BASE
    # T_base = 0.5*m_B*||W_v_B||² + 0.5*B_ω_B^T*I_B*B_ω_B
    # ========================================
    T_base_trans = 0.5 * m_base * np.dot(v_WB, v_WB)
    T_base_rot = 0.5 * np.dot(B_omega_B, I_base @ B_omega_B)
    T_base = T_base_trans + T_base_rot
    
    # ========================================
    # 2. ENERGIA CINETICA BRACCI + 
    # 3. ENERGIA POTENZIALE MOLLE TORSIONALI
    # ========================================
    T_arms_trans = 0.0
    T_arms_rot = 0.0
    V_spring_total = 0.0
    z_arms = []  # Per energia potenziale gravitazionale
    
    for i in range(4):
        # Trasformazioni dalla configurazione
        T_BH = T_BH_list[i]
        T_HP = T_HP_list[i]
        
        # R_BH0: rotazione di riposo body -> hinge
        R_BH0 = T_BH[:3, :3]
        B_r_BH = T_BH[:3, 3]  # Offset body -> hinge (in body frame)
        
        # R_HP: rotazione hinge -> prop (fissa)
        R_HP = T_HP[:3, :3]
        H_r_HP = T_HP[:3, 3]  # Offset hinge -> COM braccio (in hinge frame)
        
        # Quaternione del braccio: R_H0H (rotazione relativa dal riposo)
        R_H0H = quat_to_rotmat(q_arms[i])
        
        # Rotazione corrente: R_BH = R_BH0 * R_H0H
        R_BH = R_BH0 @ R_H0H
        
        # Rotazione world -> hinge: R_WH = R_WB * R_BH
        R_WH = R_WB @ R_BH
        
        # Rotazione world -> prop: R_WP = R_WH * R_HP
        R_WP = R_WH @ R_HP
        
        # ----------------------------------------
        # Velocità angolare assoluta del braccio i
        # P_ω_P = P_ω_rel + R_PB * B_ω_B
        # ----------------------------------------
        R_PW = R_WP.T
        R_PB = R_PW @ R_WB
        P_omega_P = P_omega_rel[i] + R_PB @ B_omega_B
        W_omega_P = R_WP @ P_omega_P
        
        # ----------------------------------------
        # Velocità lineare del COM del braccio i
        # W_v_P = W_v_B + W_ω_B × W_r_BH + W_ω_P × W_r_HP
        # ----------------------------------------
        W_r_BH = R_WB @ B_r_BH  # Vettore base->hinge in world
        W_r_HP = R_WH @ H_r_HP  # Vettore hinge->prop COM in world
        
        W_v_P = v_WB + np.cross(W_omega_B, W_r_BH) + np.cross(W_omega_P, W_r_HP)
        
        # Posizione del COM del braccio in world
        W_r_P = p_WB + W_r_BH + W_r_HP
        z_arms.append(W_r_P[2])
        
        # ----------------------------------------
        # Energia cinetica braccio i
        # T_i = 0.5*m_i*||W_v_P||² + 0.5*P_ω_P^T*I_P*P_ω_P
        # ----------------------------------------
        T_arms_trans += 0.5 * m_arm * np.dot(W_v_P, W_v_P)
        T_arms_rot += 0.5 * np.dot(P_omega_P, I_arm @ P_omega_P)
        
        # ----------------------------------------
        # Energia potenziale molla torsionale
        # e_i = Log(R_err) = Log(R_BH0^T * R_BH) = Log(R_H0H)
        # V_spring,i = 0.5*k*||e_i||²
        # ----------------------------------------
        R_err = R_H0H  # R_BH0^T * R_BH = R_BH0^T * (R_BH0 * R_H0H) = R_H0H
        e_i = logSO3(R_err)
        V_spring_total += 0.5 * k_joint * np.dot(e_i, e_i)
    
    # ========================================
    # 4. ENERGIA POTENZIALE GRAVITAZIONALE
    # V_grav = Σ_j m_j * g * z_j
    # ========================================
    V_grav_base = m_base * g * p_WB[2]
    V_grav_arms = sum(m_arm * g * z for z in z_arms)
    V_grav = V_grav_base + V_grav_arms
    
    # ========================================
    # ENERGIA TOTALE
    # ========================================
    T_total = T_base + T_arms_trans + T_arms_rot
    E_total = T_total + V_grav + V_spring_total
    
    return {
        'time': sim_time,
        # Componenti energia cinetica
        'T_base': T_base,
        'T_base_trans': T_base_trans,
        'T_base_rot': T_base_rot,
        'T_arms_trans': T_arms_trans,
        'T_arms_rot': T_arms_rot,
        'T_total': T_total,
        # Componenti energia potenziale
        'V_grav': V_grav,
        'V_grav_base': V_grav_base,
        'V_grav_arms': V_grav_arms,
        'V_spring': V_spring_total,
        # Energia totale
        'E_total': E_total,
        # Debug
        'z_base': p_WB[2],
        'z_arms': z_arms,
        # Per compatibilità con vecchio formato
        'E_k_trans': T_base_trans + T_arms_trans,
        'E_k_rot_base': T_base_rot,
        'E_k_rot_arms': T_arms_rot,
        'E_k_total': T_total,
        'E_p': V_grav + V_spring_total
    }

def main():
    print("=" * 60)
    print("Energy Analysis of Multi-Body Drone System")
    print("=" * 60)
    print("\nModello: 5 corpi rigidi (base + 4 bracci)")
    print("Giunti sferici con molla torsionale + smorzatore")
    print("E_tot = T_base + Σ T_arms + V_grav + Σ V_spring")
    
    # Carica i parametri
    try:
        params = load_parameters()
        print("\n✓ Parametri caricati da model/drone_parameters.yaml")
    except FileNotFoundError:
        print("✗ Errore: model/drone_parameters.yaml non trovato")
        return
    
    # Carica le trasformazioni T_BH e T_HP dal YAML
    T_BH_list = []
    T_HP_list = []
    
    # Parse T_BH (base -> hinge)
    t_bh_data = params['transforms']['T_BH']
    for key in ['H0', 'H1', 'H2', 'H3']:
        T_BH_list.append(parse_transform(t_bh_data[key]))
    
    # Parse T_HP (hinge -> prop)
    t_hp_data = params['transforms']['T_HP']
    for key in ['H0_to_motor_0', 'H1_to_motor_1', 'H2_to_motor_2', 'H3_to_motor_3']:
        T_HP_list.append(parse_transform(t_hp_data[key]))
    
    print(f"  k_joint (rigidezza molla): {params['morphing_joint']['k_joint']}")
    print(f"  b_joint (smorzamento): {params['morphing_joint']['b_joint']}")
    
    # Trova tutti i frame JSON nella cartella export
    frame_files = sorted(glob.glob("export/frame_*.json"))
    
    if not frame_files:
        print("✗ Nessun file frame trovato in export/")
        return
    
    print(f"✓ Trovati {len(frame_files)} frame")
    
    # Calcola l'energia per ogni frame
    energies = []
    for frame_file in frame_files:
        try:
            frame_data = load_frame(frame_file)
            
            # Verifica che il frame abbia lo stato
            if 'state' not in frame_data:
                print(f"⚠ Avviso: {Path(frame_file).name} non contiene lo stato")
                continue
            
            energy = calculate_energy_from_frame(frame_data, params, T_BH_list, T_HP_list)
            energies.append(energy)
        except Exception as e:
            print(f"✗ Errore nel processare {Path(frame_file).name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not energies:
        print("✗ Nessun frame con stato valido trovato")
        return
    
    # Converte in array per plottare
    times = np.array([e['time'] for e in energies])
    
    # Componenti energia cinetica (dettagliate)
    T_base = np.array([e['T_base'] for e in energies])
    T_base_trans = np.array([e['T_base_trans'] for e in energies])
    T_base_rot = np.array([e['T_base_rot'] for e in energies])
    T_arms_trans = np.array([e['T_arms_trans'] for e in energies])
    T_arms_rot = np.array([e['T_arms_rot'] for e in energies])
    T_total = np.array([e['T_total'] for e in energies])
    
    # Componenti energia potenziale
    V_grav = np.array([e['V_grav'] for e in energies])
    V_grav_base = np.array([e['V_grav_base'] for e in energies])
    V_grav_arms = np.array([e['V_grav_arms'] for e in energies])
    V_spring = np.array([e['V_spring'] for e in energies])
    
    # Energia totale
    E_total = np.array([e['E_total'] for e in energies])
    
    z_base = np.array([e['z_base'] for e in energies])
    
    # Per compatibilità
    E_k_trans = T_base_trans + T_arms_trans
    E_k_rot_base = T_base_rot
    E_k_rot_arms = T_arms_rot
    E_k_total = T_total
    E_p = V_grav + V_spring
    
    # Stampa statistiche
    print("\n" + "=" * 60)
    print("STATISTICHE ENERGIA")
    print("=" * 60)
    
    print(f"\nTempo simulazione: {times[0]:.4f} s → {times[-1]:.4f} s")
    print(f"Numero di frame: {len(times)}")
    
    print(f"\n--- ENERGIA CINETICA ---")
    print(f"\nBase - Traslazionale:")
    print(f"  Min: {T_base_trans.min():.6e} J, Max: {T_base_trans.max():.6e} J")
    
    print(f"\nBase - Rotazionale:")
    print(f"  Min: {T_base_rot.min():.6e} J, Max: {T_base_rot.max():.6e} J")
    
    print(f"\nBracci - Traslazionale:")
    print(f"  Min: {T_arms_trans.min():.6e} J, Max: {T_arms_trans.max():.6e} J")
    
    print(f"\nBracci - Rotazionale:")
    print(f"  Min: {T_arms_rot.min():.6e} J, Max: {T_arms_rot.max():.6e} J")
    
    print(f"\nCinetica TOTALE:")
    print(f"  Min: {T_total.min():.6e} J, Max: {T_total.max():.6e} J")
    
    print(f"\n--- ENERGIA POTENZIALE ---")
    print(f"\nGravitazionale Base:")
    print(f"  Min: {V_grav_base.min():.6e} J, Max: {V_grav_base.max():.6e} J")
    
    print(f"\nGravitazionale Bracci:")
    print(f"  Min: {V_grav_arms.min():.6e} J, Max: {V_grav_arms.max():.6e} J")
    
    print(f"\nGravitazionale TOTALE:")
    print(f"  Min: {V_grav.min():.6e} J, Max: {V_grav.max():.6e} J")
    
    print(f"\nMolle Torsionali (V_spring):")
    print(f"  Min: {V_spring.min():.6e} J, Max: {V_spring.max():.6e} J")
    
    print(f"\nPotenziale TOTALE (grav + spring):")
    print(f"  Min: {E_p.min():.6e} J, Max: {E_p.max():.6e} J")
    
    print(f"\n--- ENERGIA TOTALE ---")
    print(f"\nE_tot = T_base + Σ T_arms + V_grav + Σ V_spring:")
    print(f"  Iniziale: {E_total[0]:.6e} J")
    print(f"  Finale: {E_total[-1]:.6e} J")
    print(f"  Min: {E_total.min():.6e} J")
    print(f"  Max: {E_total.max():.6e} J")
    print(f"  Variazione: {E_total.max() - E_total.min():.6e} J")
    
    # Calcola la percentuale di variazione
    E_total_range = E_total.max() - E_total.min()
    E_total_mean = E_total.mean()
    if E_total_mean != 0:
        percent_variation = (E_total_range / abs(E_total_mean)) * 100
        print(f"  Variazione relativa: {percent_variation:.4f}%")
    
    # Variazione rispetto al valore iniziale
    delta_E = E_total[-1] - E_total[0]
    if E_total[0] != 0:
        percent_dissipation = (delta_E / abs(E_total[0])) * 100
        print(f"  Dissipazione (finale - iniziale): {delta_E:.6e} J ({percent_dissipation:.4f}%)")
    
    print(f"\nAltezza base:")
    print(f"  Min: {z_base.min():.6f} m")
    print(f"  Max: {z_base.max():.6f} m")
    
    # Crea i plot
    print("\n" + "=" * 60)
    print("Creazione grafici...")
    print("=" * 60)
    
    # ========================================
    # FIGURA 1: Pannello 3x2 con tutte le componenti
    # ========================================
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # Plot 1: Energia cinetica traslazionale (base + bracci)
    axes[0, 0].plot(times, T_base_trans, 'b-', label='Base', linewidth=1.5)
    axes[0, 0].plot(times, T_arms_trans, 'c--', label='Bracci', linewidth=1.5)
    axes[0, 0].set_ylabel('Energia (J)')
    axes[0, 0].set_title('Energia Cinetica Traslazionale')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Plot 2: Energia cinetica rotazionale
    axes[0, 1].plot(times, T_base_rot, 'g-', label='Base', linewidth=1.5)
    axes[0, 1].plot(times, T_arms_rot, 'r--', label='Bracci', linewidth=1.5)
    axes[0, 1].set_ylabel('Energia (J)')
    axes[0, 1].set_title('Energia Cinetica Rotazionale')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Plot 3: Energia cinetica totale
    axes[1, 0].plot(times, T_total, 'purple', linewidth=1.5)
    axes[1, 0].set_ylabel('Energia (J)')
    axes[1, 0].set_title('Energia Cinetica Totale (T)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Plot 4: Energia potenziale (gravitazionale + molle)
    axes[1, 1].plot(times, V_grav, 'orange', label='Gravitazionale', linewidth=1.5)
    axes[1, 1].plot(times, V_spring, 'm--', label='Molle torsionali', linewidth=1.5)
    axes[1, 1].set_ylabel('Energia (J)')
    axes[1, 1].set_title('Energia Potenziale (V_grav + V_spring)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Plot 5: Energia totale (cinetica + potenziale)
    axes[2, 0].plot(times, E_total, 'k-', linewidth=1.5, label='E_totale')
    axes[2, 0].axhline(y=E_total[0], color='red', linestyle='--', alpha=0.7, label='E_iniziale')
    axes[2, 0].set_xlabel('Tempo (s)')
    axes[2, 0].set_ylabel('Energia (J)')
    axes[2, 0].set_title('Energia Totale del Sistema')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Plot 6: Altezza del base + energia molle
    ax1 = axes[2, 1]
    ax2 = ax1.twinx()
    l1 = ax1.plot(times, z_base, 'cyan', linewidth=1.5, label='Altezza base')
    l2 = ax2.plot(times, V_spring, 'm--', linewidth=1.5, label='V_spring')
    ax1.set_xlabel('Tempo (s)')
    ax1.set_ylabel('Altezza (m)', color='cyan')
    ax2.set_ylabel('V_spring (J)', color='m')
    ax1.set_title('Altezza Base e Energia Molle')
    ax1.grid(True, alpha=0.3)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')
    
    plt.tight_layout()
    plt.savefig('energy_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Grafico salvato come 'energy_analysis.png'")
    
    # ========================================
    # FIGURA 2: Confronto tutte le componenti
    fig2, ax = plt.subplots(figsize=(14, 7))
    
    # Energie cinetiche
    ax.plot(times, T_base_trans + T_arms_trans, 'b-', label='T traslazionale (base+bracci)', linewidth=2)
    ax.plot(times, T_base_rot, 'g-', label='T rot base', linewidth=2)
    ax.plot(times, T_arms_rot, 'r-', label='T rot bracci', linewidth=2)
    
    # Energie potenziali
    ax.plot(times, V_grav, 'orange', label='V gravitazionale', linewidth=2)
    ax.plot(times, V_spring, 'm-', label='V molle torsionali', linewidth=2)
    
    # Totale
    ax.plot(times, E_total, 'k--', label='E_totale', linewidth=2.5)
    
    ax.set_xlabel('Tempo (s)', fontsize=12)
    ax.set_ylabel('Energia (J)', fontsize=12)
    ax.set_title('Componenti di Energia nel Tempo\n' + 
                 r'$E_{tot} = T_{base} + \Sigma T_{arms} + V_{grav} + \Sigma V_{spring}$', fontsize=14)
    ax.legend(fontsize=10, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig('energy_components.png', dpi=150, bbox_inches='tight')
    print("✓ Grafico componenti salvato come 'energy_components.png'")
    
    # ========================================
    # FIGURA 3: Conservazione energia
    fig3, axes3 = plt.subplots(2, 1, figsize=(12, 8))
    
    # Pannello superiore: energia totale assoluta
    axes3[0].plot(times, E_total, 'k-', linewidth=2, label='E_totale')
    axes3[0].axhline(y=E_total[0], color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='E_iniziale')
    axes3[0].fill_between(times, E_total.min(), E_total.max(), alpha=0.1, color='gray')
    axes3[0].set_ylabel('Energia (J)', fontsize=12)
    axes3[0].set_title('Energia Totale del Sistema (senza smorzamento dissipativo)', fontsize=14)
    axes3[0].legend(fontsize=10)
    axes3[0].grid(True, alpha=0.3)
    axes3[0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Pannello inferiore: variazione relativa
    # Normalizza l'energia rispetto al valore iniziale
    if abs(E_total[0]) > 1e-12:
        E_total_normalized = (E_total - E_total[0]) / abs(E_total[0]) * 100
    else:
        E_total_normalized = E_total - E_total[0]
    
    axes3[1].plot(times, E_total_normalized, 'k-', linewidth=2)
    axes3[1].axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    axes3[1].fill_between(times, E_total_normalized, 0, alpha=0.2, color='blue' if E_total_normalized[-1] < 0 else 'red')
    axes3[1].set_xlabel('Tempo (s)', fontsize=12)
    axes3[1].set_ylabel('Variazione relativa (%)', fontsize=12)
    axes3[1].set_title(f'Conservazione Energia (dissipazione smorzatori: {E_total_normalized[-1]:.4f}%)', fontsize=14)
    axes3[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('energy_conservation.png', dpi=150, bbox_inches='tight')
    print("✓ Grafico conservazione salvato come 'energy_conservation.png'")
    
    print("\n" + "=" * 60)
    print("Analisi completata!")
    print("=" * 60)
    
    # Mostra i grafici
    plt.show()

if __name__ == '__main__':
    main()
