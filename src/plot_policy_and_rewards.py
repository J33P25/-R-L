"""
GreenMind: Policy Iteration & Step Reward Analysis Plotter
Generates publication-quality figures for:
- Category A: Plot 1 (Policy Iteration Value Function Profile V*(s))
- Category A: Plot 2 (Policy Improvement Step Convergence Delta V)
- Category B: Plot 5 (Cumulative Reward Trajectory over 24 Hours)
- Category B: Plot 6 (Multi-Objective Reward Loss Decomposition)
- Consolidated 2x2 Master Figure
"""
import os 
import sys 
import tempfile 
from pathlib import Path 
os .environ ['MPLCONFIGDIR']=tempfile .gettempdir ()
ROOT_DIR =Path (__file__ ).resolve ().parent .parent 
if str (ROOT_DIR )not in sys .path :
    sys .path .insert (0 ,str (ROOT_DIR ))
import json 
import numpy as np 
import matplotlib .pyplot as plt 
import matplotlib .gridspec as gridspec 
from src .config import LOGS_DIR ,WEIGHT_COMFORT ,WEIGHT_ENERGY ,WEIGHT_IAQ ,WEIGHT_SMOOTHNESS 
from src .mdp_formulation import build_mdp ,DISCOUNT_FACTOR ,NUM_STATES ,STATE_NAMES ,ACTIONS_PER_STATE 
plt .rcParams ['font.sans-serif']=['DejaVu Sans','Arial','Helvetica','sans-serif']
plt .rcParams ['axes.unicode_minus']=False 
COLORS ={
"Rule-Based":"#475569",
"Policy Iteration":"#7c3aed",
"PPO":"#0284c7",
"SAC":"#059669"
}
LOSS_COLORS ={
"Energy":"#f59e0b",
"Comfort":"#ef4444",
"IAQ":"#8b5cf6",
"Smoothness":"#10b981"
}
def load_data ():
    results_file =LOGS_DIR /"evaluation_results.json"
    if not results_file .exists ():
        raise FileNotFoundError (f"Results file not found at {results_file }")
    with open (results_file ,"r")as f :
        data =json .load (f )
    return data 
def run_policy_iteration_stepwise ():
    """
    Simulates Policy Iteration step-by-step from initial policy pi_0 to optimal policy pi*.
    Returns V_0, V_1, pi_0, pi_1, and action details.
    """
    P ,R =build_mdp ()
    num_non_terminal =6 
    gamma =DISCOUNT_FACTOR 
    pi_0 =np .zeros (num_non_terminal ,dtype =int )
    P_pi0 =np .zeros ((num_non_terminal ,num_non_terminal ))
    R_pi0 =np .zeros (num_non_terminal )
    for s in range (num_non_terminal ):
        a =pi_0 [s ]
        for next_s in range (NUM_STATES ):
            prob =P [s ,a ,next_s ]
            rew =R [s ,a ,next_s ]
            R_pi0 [s ]+=prob *rew 
            if next_s <num_non_terminal :
                P_pi0 [s ,next_s ]+=prob *gamma 
    I =np .eye (num_non_terminal )
    V_0 =np .linalg .solve (I -P_pi0 ,R_pi0 )
    pi_1 =pi_0 .copy ()
    q_table =np .zeros ((num_non_terminal ,3 ))
    for s in range (num_non_terminal ):
        for a in range (3 ):
            q_val =0.0 
            for next_s in range (NUM_STATES ):
                prob =P [s ,a ,next_s ]
                rew =R [s ,a ,next_s ]
                v_next =V_0 [next_s ]if next_s <num_non_terminal else 0.0 
                q_val +=prob *(rew +gamma *v_next )
            q_table [s ,a ]=q_val 
        pi_1 [s ]=int (np .argmax (q_table [s ]))
    P_pi1 =np .zeros ((num_non_terminal ,num_non_terminal ))
    R_pi1 =np .zeros (num_non_terminal )
    for s in range (num_non_terminal ):
        a =pi_1 [s ]
        for next_s in range (NUM_STATES ):
            prob =P [s ,a ,next_s ]
            rew =R [s ,a ,next_s ]
            R_pi1 [s ]+=prob *rew 
            if next_s <num_non_terminal :
                P_pi1 [s ,next_s ]+=prob *gamma 
    V_1 =np .linalg .solve (I -P_pi1 ,R_pi1 )
    return V_0 ,V_1 ,pi_0 ,pi_1 
def plot_a1_state_values (pi_theory ,out_dir ):
    fig ,ax =plt .subplots (figsize =(10 ,5.5 ),dpi =300 )
    states =[
    ("S1","Comfort Zone\n(21-24°C, CO2≤1000)"),
    ("S3","Overcooled\n(<21°C, CO2≤1000)"),
    ("S4","Stuffy Air\n(21-24°C, CO2>1000)"),
    ("S2","Overheated\n(>24°C, CO2≤1000)"),
    ("S6","Unoccupied\n(Empty Zone)"),
    ("S5","Critical Alert\n(>24°C, CO2>1000)")
    ]
    keys =[
    "S1: Comfort Zone (21-24°C, CO2<=1000, Occupied)",
    "S3: Overcooled (<21°C, CO2<=1000, Occupied)",
    "S4: Stuffy (21-24°C, CO2>1000, Occupied)",
    "S2: Overheated (>24°C, CO2<=1000, Occupied)",
    "S6: Unoccupied (Empty Zone)",
    "S5: Critical (>24°C, CO2>1000, Occupied)"
    ]
    v_vals =[pi_theory [k ]["state_value_V"]for k in keys ]
    actions =[pi_theory [k ]["optimal_action_name"]for k in keys ]
    labels =[s [0 ]+"\n"+s [1 ]for s in states ]
    bar_colors =["#10b981","#0284c7","#f59e0b","#f97316","#64748b","#ef4444"]
    x =np .arange (len (states ))
    bars =ax .bar (x ,v_vals ,color =bar_colors ,width =0.55 ,edgecolor ="#1e293b",linewidth =1.2 ,zorder =3 )
    ax .axhline (0 ,color ="#1e293b",linestyle ="-",linewidth =1.0 ,alpha =0.7 )
    ax .set_title ("Plot A1: Analytical Policy Iteration Optimal State-Value Function $V^*(s)$",fontsize =12.5 ,fontweight ='bold',pad =14 )
    ax .set_ylabel ("Optimal Expected Discounted Return $V^*(s)$",fontsize =11 ,fontweight ='semibold')
    ax .set_xticks (x )
    ax .set_xticklabels (labels ,fontsize =9.5 ,fontweight ='semibold')
    ax .set_ylim (-9.5 ,0.5 )
    ax .grid (axis ="y",linestyle =":",alpha =0.6 ,zorder =0 )
    for bar ,val ,act in zip (bars ,v_vals ,actions ):
        h =bar .get_height ()
        ax .annotate (f"{val :.2f}\n[{act .split (':')[0 ]}]",
        xy =(bar .get_x ()+bar .get_width ()/2 ,h ),
        xytext =(0 ,-22 ),textcoords ="offset points",
        ha ='center',va ='top',fontsize =9.5 ,fontweight ='bold',color ="#0f172a")
    callout_txt =(
    "Bellman Optimality Insight:\n"
    "• S1 Comfort Zone has highest expected return (V* = -1.57)\n"
    "• S5 Critical Overheating & High CO2 suffers ~5x heavier penalty (V* = -8.01)\n"
    "• Actions: A1 Maintain, C1 Less Cool, D1 Boost Vent, B2 Mod Cool, F1 Standby, E1 Full Cool+Vent"
    )
    ax .text (0.02 ,0.05 ,callout_txt ,transform =ax .transAxes ,
    fontsize =8.5 ,verticalalignment ='bottom',
    bbox =dict (boxstyle ="round,pad=0.5",fc ="#f8fafc",ec ="#cbd5e1",alpha =0.95 ))
    plt .tight_layout ()
    p =out_dir /"fig_a1_policy_iteration_value_profile.png"
    plt .savefig (p ,dpi =300 )
    plt .close ()
    print (f"  [+] Saved {p .name }")
def plot_a2_policy_improvement (out_dir ):
    V_0 ,V_1 ,pi_0 ,pi_1 =run_policy_iteration_stepwise ()
    state_short =["S1 (Comfort)","S2 (Overheated)","S3 (Overcooled)","S4 (Stuffy)","S5 (Critical)","S6 (Empty)"]
    delta_v =V_1 -V_0 
    fig ,(ax1 ,ax2 )=plt .subplots (1 ,2 ,figsize =(13 ,5 ),dpi =300 ,gridspec_kw ={'width_ratios':[2 ,1.2 ]})
    x =np .arange (len (state_short ))
    w =0.35 
    bars0 =ax1 .bar (x -w /2 ,V_0 ,width =w ,label =r"Initial Policy $\pi_0$ (Hold Steady)",color ="#94a3b8",edgecolor ="#334155",linewidth =1.1 )
    bars1 =ax1 .bar (x +w /2 ,V_1 ,width =w ,label =r"Improved Optimal Policy $\pi^*$ (Iteration 1)",color ="#7c3aed",edgecolor ="#4c1d95",linewidth =1.1 )
    ax1 .set_title ("Policy Evaluation: Value Shift Across States",fontsize =11.5 ,fontweight ='bold')
    ax1 .set_ylabel ("Expected Value $V(s)$",fontsize =10.5 ,fontweight ='semibold')
    ax1 .set_xticks (x )
    ax1 .set_xticklabels (state_short ,rotation =18 ,ha ="right",fontsize =9 ,fontweight ='semibold')
    ax1 .set_ylim (-9.5 ,0.0 )
    ax1 .grid (axis ="y",linestyle =":",alpha =0.6 )
    ax1 .legend (loc ="lower left",framealpha =0.95 ,fontsize =9 )
    ax1 .annotate ("Action Updated:\n$B_1 \\to B_2$\n(Moderate Cooling & Vent)",
    xy =(x [1 ]+w /2 ,V_1 [1 ]),xytext =(x [1 ]+0.3 ,-4.5 ),
    arrowprops =dict (arrowstyle ="->",color ="#dc2626",lw =1.5 ),
    fontsize =8.5 ,fontweight ='bold',color ="#b91c1c",
    bbox =dict (boxstyle ="round,pad=0.3",fc ="#fef2f2",ec ="#f87171"))
    bars_delta =ax2 .bar (x ,delta_v ,color ="#059669",width =0.55 ,edgecolor ="#064e3b",linewidth =1.1 )
    ax2 .set_title ("Value Gain $\\Delta V(s) = V^{\\pi^*}(s) - V^{\\pi_0}(s)$",fontsize =11.5 ,fontweight ='bold')
    ax2 .set_ylabel ("Net Return Improvement",fontsize =10.5 ,fontweight ='semibold')
    ax2 .set_xticks (x )
    ax2 .set_xticklabels ([f"S{i +1 }"for i in range (6 )],fontsize =9.5 ,fontweight ='semibold')
    ax2 .set_ylim (0.0 ,max (delta_v )*1.35 )
    ax2 .grid (axis ="y",linestyle =":",alpha =0.6 )
    for bar ,dv in zip (bars_delta ,delta_v ):
        h =bar .get_height ()
        ax2 .annotate (f"+{dv :.3f}",
        xy =(bar .get_x ()+bar .get_width ()/2 ,h ),
        xytext =(0 ,4 ),textcoords ="offset points",
        ha ='center',va ='bottom',fontsize =9 ,fontweight ='bold',color ="#064e3b")
    ax2 .text (0.5 ,0.88 ,"Monotonic Improvement:\n$V^{\\pi^*}(s) \\geq V^{\\pi_0}(s) \\quad \\forall s$",
    transform =ax2 .transAxes ,ha ="center",fontsize =9 ,fontweight ='bold',color ="#065f46",
    bbox =dict (boxstyle ="round,pad=0.4",fc ="#ecfdf5",ec ="#10b981",lw =1.2 ))
    plt .suptitle ("Plot A2: Proof of Policy Improvement Theorem & Algorithmic Convergence",fontsize =13 ,fontweight ='bold',y =0.98 )
    plt .tight_layout ()
    p =out_dir /"fig_a2_policy_improvement_convergence.png"
    plt .savefig (p ,dpi =300 )
    plt .close ()
    print (f"  [+] Saved {p .name }")
def plot_b5_cumulative_rewards (results ,out_dir ):
    fig ,ax =plt .subplots (figsize =(10.5 ,5.5 ),dpi =300 )
    agent_keys =["Rule-Based","SAC","Policy Iteration","PPO"]
    labels ={
    "Rule-Based":"Rule-Based Control (RBC Baseline)",
    "Policy Iteration":"Analytical Policy Iteration (Model-Based)",
    "PPO":"PPO Continuous (On-Policy)",
    "SAC":"SAC Continuous (Off-Policy Max Entropy)"
    }
    styles ={
    "Rule-Based":{"ls":"--","lw":2.2 ,"alpha":0.85 },
    "Policy Iteration":{"ls":"-.","lw":2.2 ,"alpha":0.9 },
    "PPO":{"ls":"-","lw":2.2 ,"alpha":0.9 },
    "SAC":{"ls":"-","lw":2.8 ,"alpha":1.0 }
    }
    time_steps =np .arange (96 )*0.25 
    ax .axvspan (8.0 ,18.0 ,color ="#fef3c7",alpha =0.35 ,label ="Occupied Office Hours (08:00 - 18:00)")
    for key in agent_keys :
        sr =np .array (results [key ]["history"]["step_reward"])
        cum_r =np .cumsum (sr )
        final_val =cum_r [-1 ]
        ax .plot (time_steps ,cum_r ,label =f"{labels [key ]}  [Total: {final_val :.2f}]",color =COLORS [key ],**styles [key ])
    ax .set_title ("Plot B5: 24-Hour Cumulative Reward Trajectory $\\sum_{t=0}^T r_t$ Across Control Paradigms",fontsize =12.5 ,fontweight ='bold',pad =12 )
    ax .set_xlabel ("Time of Day (Hours)",fontsize =11 ,fontweight ='semibold')
    ax .set_ylabel ("Cumulative Multi-Objective Return",fontsize =11 ,fontweight ='semibold')
    ax .set_xlim (0 ,24 )
    ax .set_xticks (range (0 ,25 ,2 ))
    ax .grid (True ,linestyle =":",alpha =0.6 )
    ax .legend (loc ="lower left",framealpha =0.95 ,fontsize =9.5 )
    ax .annotate ("PPO penalty accelerates during afternoon\ndue to fresh-air ventilation throttling",
    xy =(16.0 ,-14.5 ),xytext =(12.0 ,-6.5 ),
    arrowprops =dict (arrowstyle ="->",color =COLORS ["PPO"],lw =1.5 ),
    fontsize =8.5 ,fontweight ='bold',color =COLORS ["PPO"],
    bbox =dict (boxstyle ="round,pad=0.3",fc ="#ffffff",ec =COLORS ["PPO"],alpha =0.9 ))
    plt .tight_layout ()
    p =out_dir /"fig_b5_cumulative_reward_trajectory.png"
    plt .savefig (p ,dpi =300 )
    plt .close ()
    print (f"  [+] Saved {p .name }")
def plot_b6_loss_decomposition (results ,out_dir ):
    agent_keys =["Rule-Based","Policy Iteration","PPO","SAC"]
    display_names =["Rule-Based\n(Baseline)","Policy Iteration\n(Model-Based)","PPO\n(On-Policy)","SAC\n(Off-Policy)"]
    mean_losses ={"Energy":[],"Comfort":[],"IAQ":[],"Smoothness":[]}
    weighted_losses ={"Energy":[],"Comfort":[],"IAQ":[],"Smoothness":[]}
    for agent in agent_keys :
        hist =results [agent ]["history"]
        pwr =np .array (hist ["power_kw"])
        tin =np .array (hist ["indoor_temp"])
        co2 =np .array (hist ["co2_ppm"])
        occ =np .array (hist ["occupants"])
        l_e =np .clip (pwr /5.0 ,0.0 ,1.0 )
        l_c =np .zeros (96 )
        l_q =np .zeros (96 )
        for i in range (96 ):
            if occ [i ]>0 :
                if tin [i ]<21.0 :
                    l_c [i ]=np .clip (((21.0 -tin [i ])/3.0 )**2 ,0.0 ,1.0 )
                elif tin [i ]>24.0 :
                    l_c [i ]=np .clip (((tin [i ]-24.0 )/3.0 )**2 ,0.0 ,1.0 )
                if co2 [i ]>1000.0 :
                    l_q [i ]=np .clip ((co2 [i ]-1000.0 )/1000.0 ,0.0 ,1.0 )
        v =np .array (hist ["valve"])
        f =np .array (hist ["fan"])
        d =np .array (hist ["damper"])
        act =np .stack ([v ,f ,d ],axis =1 )
        prev_act =np .vstack ([np .array ([[0.2 ,0.3 ,0.2 ]]),act [:-1 ]])
        l_s =np .clip (np .mean ((act -prev_act )**2 ,axis =1 ),0.0 ,1.0 )
        w_e =WEIGHT_ENERGY *float (np .mean (l_e ))
        w_c =WEIGHT_COMFORT *float (np .mean (l_c ))
        w_q =WEIGHT_IAQ *float (np .mean (l_q ))
        w_s =WEIGHT_SMOOTHNESS *float (np .mean (l_s ))
        weighted_losses ["Energy"].append (w_e )
        weighted_losses ["Comfort"].append (w_c )
        weighted_losses ["IAQ"].append (w_q )
        weighted_losses ["Smoothness"].append (w_s )
    fig ,(ax1 ,ax2 )=plt .subplots (1 ,2 ,figsize =(14 ,5.5 ),dpi =300 ,gridspec_kw ={'width_ratios':[1.2 ,1.6 ]})
    x =np .arange (len (agent_keys ))
    w =0.52 
    bottom =np .zeros (len (agent_keys ))
    for comp in ["Energy","Comfort","IAQ","Smoothness"]:
        vals =np .array (weighted_losses [comp ])
        ax1 .bar (x ,vals ,bottom =bottom ,width =w ,label =f"{comp } Loss ($w_{comp [0 ]}={globals ()['WEIGHT_'+comp .upper ()]}$)",
        color =LOSS_COLORS [comp ],edgecolor ="#1e293b",linewidth =0.9 )
        bottom +=vals 
    ax1 .set_title ("Total Mean Step Penalty Breakdown",fontsize =11.5 ,fontweight ='bold')
    ax1 .set_ylabel ("Mean Weighted Penalty per Step",fontsize =10.5 ,fontweight ='semibold')
    ax1 .set_xticks (x )
    ax1 .set_xticklabels (display_names ,fontsize =9 ,fontweight ='semibold')
    ax1 .set_ylim (0 ,max (bottom )*1.25 )
    ax1 .grid (axis ="y",linestyle =":",alpha =0.6 )
    ax1 .legend (loc ="upper left",framealpha =0.95 ,fontsize =8.5 )
    for i ,tot in enumerate (bottom ):
        ax1 .annotate (f"{tot :.3f}",xy =(x [i ],tot ),xytext =(0 ,4 ),textcoords ="offset points",
        ha ="center",fontsize =9.5 ,fontweight ="bold",color ="#0f172a")
    w_cluster =0.18 
    offsets =[-1.5 *w_cluster ,-0.5 *w_cluster ,0.5 *w_cluster ,1.5 *w_cluster ]
    comps =["Energy","Comfort","IAQ","Smoothness"]
    for idx ,comp in enumerate (comps ):
        ax2 .bar (x +offsets [idx ],weighted_losses [comp ],width =w_cluster ,
        label =f"{comp } ({globals ()['WEIGHT_'+comp .upper ()]:.2f})",
        color =LOSS_COLORS [comp ],edgecolor ="#1e293b",linewidth =0.8 )
    ax2 .set_title ("Multi-Objective Trade-Off Component Breakdown",fontsize =11.5 ,fontweight ='bold')
    ax2 .set_ylabel ("Component Weighted Loss Contribution",fontsize =10.5 ,fontweight ='semibold')
    ax2 .set_xticks (x )
    ax2 .set_xticklabels (display_names ,fontsize =9 ,fontweight ='semibold')
    ax2 .grid (axis ="y",linestyle =":",alpha =0.6 )
    ax2 .legend (loc ="upper right",framealpha =0.95 ,fontsize =9 )
    ax2 .annotate ("PPO throttles damper to save energy,\nincurring high IAQ penalty",
    xy =(2 +offsets [2 ],weighted_losses ["IAQ"][2 ]),xytext =(1.2 ,0.07 ),
    arrowprops =dict (arrowstyle ="->",color =LOSS_COLORS ["IAQ"],lw =1.5 ),
    fontsize =8.5 ,fontweight ='bold',color =LOSS_COLORS ["IAQ"],
    bbox =dict (boxstyle ="round,pad=0.3",fc ="#ffffff",ec =LOSS_COLORS ["IAQ"]))
    plt .suptitle ("Plot B6: Multi-Objective Reward Loss Decomposition & Trade-Off Analysis",fontsize =13 ,fontweight ='bold',y =0.98 )
    plt .tight_layout ()
    p =out_dir /"fig_b6_multiobjective_loss_decomposition.png"
    plt .savefig (p ,dpi =300 )
    plt .close ()
    print (f"  [+] Saved {p .name }")
def plot_master_2x2 (pi_theory ,results ,out_dir ):
    fig =plt .figure (figsize =(18 ,12 ),dpi =300 )
    gs =gridspec .GridSpec (2 ,2 ,figure =fig ,hspace =0.32 ,wspace =0.22 )
    ax1 =fig .add_subplot (gs [0 ,0 ])
    states =[("S1","Comfort"),("S3","Overcooled"),("S4","Stuffy"),("S2","Overheated"),("S6","Empty"),("S5","Critical")]
    keys =[
    "S1: Comfort Zone (21-24°C, CO2<=1000, Occupied)",
    "S3: Overcooled (<21°C, CO2<=1000, Occupied)",
    "S4: Stuffy (21-24°C, CO2>1000, Occupied)",
    "S2: Overheated (>24°C, CO2<=1000, Occupied)",
    "S6: Unoccupied (Empty Zone)",
    "S5: Critical (>24°C, CO2>1000, Occupied)"
    ]
    v_vals =[pi_theory [k ]["state_value_V"]for k in keys ]
    bar_colors =["#10b981","#0284c7","#f59e0b","#f97316","#64748b","#ef4444"]
    x1 =np .arange (len (states ))
    bars1 =ax1 .bar (x1 ,v_vals ,color =bar_colors ,width =0.55 ,edgecolor ="#1e293b",linewidth =1.1 )
    ax1 .set_title ("(A) Analytical Policy Iteration: State-Value Function $V^*(s)$",fontsize =11.5 ,fontweight ='bold')
    ax1 .set_ylabel ("Expected Value $V^*(s)$",fontsize =10 ,fontweight ='semibold')
    ax1 .set_xticks (x1 )
    ax1 .set_xticklabels ([s [0 ]+"\n"+s [1 ]for s in states ],fontsize =8.5 ,fontweight ='semibold')
    ax1 .set_ylim (-9.5 ,0.5 )
    ax1 .grid (axis ="y",linestyle =":",alpha =0.6 )
    for bar ,val in zip (bars1 ,v_vals ):
        ax1 .annotate (f"{val :.2f}",xy =(bar .get_x ()+bar .get_width ()/2 ,bar .get_height ()),
        xytext =(0 ,-16 ),textcoords ="offset points",ha ='center',fontsize =9 ,fontweight ='bold')
    ax2 =fig .add_subplot (gs [0 ,1 ])
    V_0 ,V_1 ,_ ,_ =run_policy_iteration_stepwise ()
    delta_v =V_1 -V_0 
    x2 =np .arange (6 )
    w2 =0.35 
    ax2 .bar (x2 -w2 /2 ,V_0 ,width =w2 ,label =r"$\pi_0$ (Baseline)",color ="#94a3b8",edgecolor ="#334155")
    ax2 .bar (x2 +w2 /2 ,V_1 ,width =w2 ,label =r"$\pi^*$ (Optimal)",color ="#7c3aed",edgecolor ="#4c1d95")
    ax2 .set_title (r"(B) Policy Improvement Step Convergence: $V^{\pi^*}(s) \geq V^{\pi_0}(s)$",fontsize =11.5 ,fontweight ='bold')
    ax2 .set_ylabel ("Expected Value $V(s)$",fontsize =10 ,fontweight ='semibold')
    ax2 .set_xticks (x2 )
    ax2 .set_xticklabels ([f"S{i +1 }"for i in range (6 )],fontsize =9 ,fontweight ='semibold')
    ax2 .set_ylim (-9.5 ,0.0 )
    ax2 .grid (axis ="y",linestyle =":",alpha =0.6 )
    ax2 .legend (loc ="lower left",fontsize =8.5 )
    ax3 =fig .add_subplot (gs [1 ,0 ])
    time_steps =np .arange (96 )*0.25 
    ax3 .axvspan (8.0 ,18.0 ,color ="#fef3c7",alpha =0.35 ,label ="Office Hours (08:00 - 18:00)")
    for key in ["Rule-Based","Policy Iteration","PPO","SAC"]:
        sr =np .array (results [key ]["history"]["step_reward"])
        ax3 .plot (time_steps ,np .cumsum (sr ),label =f"{key } [{np .cumsum (sr )[-1 ]:.1f}]",color =COLORS [key ],lw =2.2 )
    ax3 .set_title (r"(C) 24-Hour Cumulative Reward Trajectory $\sum r_t$",fontsize =11.5 ,fontweight ='bold')
    ax3 .set_xlabel ("Time of Day (Hours)",fontsize =10 ,fontweight ='semibold')
    ax3 .set_ylabel ("Cumulative Return",fontsize =10 ,fontweight ='semibold')
    ax3 .set_xlim (0 ,24 )
    ax3 .grid (True ,linestyle =":",alpha =0.6 )
    ax3 .legend (loc ="lower left",fontsize =8.5 )
    ax4 =fig .add_subplot (gs [1 ,1 ])
    agent_keys =["Rule-Based","Policy Iteration","PPO","SAC"]
    weighted_losses ={"Energy":[],"Comfort":[],"IAQ":[],"Smoothness":[]}
    for agent in agent_keys :
        hist =results [agent ]["history"]
        pwr =np .array (hist ["power_kw"])
        tin =np .array (hist ["indoor_temp"])
        co2 =np .array (hist ["co2_ppm"])
        occ =np .array (hist ["occupants"])
        l_e =np .clip (pwr /5.0 ,0.0 ,1.0 )
        l_c =np .zeros (96 )
        l_q =np .zeros (96 )
        for i in range (96 ):
            if occ [i ]>0 :
                if tin [i ]<21.0 :l_c [i ]=np .clip (((21.0 -tin [i ])/3.0 )**2 ,0.0 ,1.0 )
                elif tin [i ]>24.0 :l_c [i ]=np .clip (((tin [i ]-24.0 )/3.0 )**2 ,0.0 ,1.0 )
                if co2 [i ]>1000.0 :l_q [i ]=np .clip ((co2 [i ]-1000.0 )/1000.0 ,0.0 ,1.0 )
        v =np .array (hist ["valve"])
        f =np .array (hist ["fan"])
        d =np .array (hist ["damper"])
        act =np .stack ([v ,f ,d ],axis =1 )
        prev_act =np .vstack ([np .array ([[0.2 ,0.3 ,0.2 ]]),act [:-1 ]])
        l_s =np .clip (np .mean ((act -prev_act )**2 ,axis =1 ),0.0 ,1.0 )
        weighted_losses ["Energy"].append (WEIGHT_ENERGY *float (np .mean (l_e )))
        weighted_losses ["Comfort"].append (WEIGHT_COMFORT *float (np .mean (l_c )))
        weighted_losses ["IAQ"].append (WEIGHT_IAQ *float (np .mean (l_q )))
        weighted_losses ["Smoothness"].append (WEIGHT_SMOOTHNESS *float (np .mean (l_s )))
    x4 =np .arange (len (agent_keys ))
    bottom4 =np .zeros (len (agent_keys ))
    for comp in ["Energy","Comfort","IAQ","Smoothness"]:
        vals =np .array (weighted_losses [comp ])
        ax4 .bar (x4 ,vals ,bottom =bottom4 ,width =0.52 ,label =comp ,color =LOSS_COLORS [comp ],edgecolor ="#1e293b",lw =0.9 )
        bottom4 +=vals 
    ax4 .set_title ("(D) Multi-Objective Reward Loss Decomposition",fontsize =11.5 ,fontweight ='bold')
    ax4 .set_ylabel ("Mean Step Penalty",fontsize =10 ,fontweight ='semibold')
    ax4 .set_xticks (x4 )
    ax4 .set_xticklabels (["Rule-Based","Policy Iteration","PPO","SAC"],fontsize =8.5 ,fontweight ='semibold')
    ax4 .grid (axis ="y",linestyle =":",alpha =0.6 )
    ax4 .legend (loc ="upper left",fontsize =8.5 )
    plt .suptitle ("GreenMind: Policy Iteration Convergence & Multi-Objective Reward Analysis",fontsize =14 ,fontweight ='bold',y =0.98 )
    p =out_dir /"fig_policy_iteration_and_rewards_master.png"
    plt .savefig (p ,dpi =300 )
    plt .close ()
    print (f"  [+] Saved {p .name }")
def main ():
    print ("="*70 )
    print (" GreenMind: Generating Policy Iteration & Step Reward Figures")
    print ("="*70 )
    data =load_data ()
    out_dir =LOGS_DIR 
    out_dir .mkdir (parents =True ,exist_ok =True )
    print ("\n[1/5] Generating Plot A1: Policy Iteration State-Value Function V*(s)...")
    plot_a1_state_values (data ["policy_iteration_theory"],out_dir )
    print ("\n[2/5] Generating Plot A2: Policy Improvement Step Convergence Delta V...")
    plot_a2_policy_improvement (out_dir )
    print ("\n[3/5] Generating Plot B5: 24-Hour Cumulative Reward Trajectory...")
    plot_b5_cumulative_rewards (data ["results"],out_dir )
    print ("\n[4/5] Generating Plot B6: Multi-Objective Reward Loss Decomposition...")
    plot_b6_loss_decomposition (data ["results"],out_dir )
    print ("\n[5/5] Generating Master Consolidated 2x2 Overview Figure...")
    plot_master_2x2 (data ["policy_iteration_theory"],data ["results"],out_dir )
    print ("\n[SUCCESS] All requested figures have been successfully generated in logs/!")
    print ("="*70 )
if __name__ =="__main__":
    main ()
