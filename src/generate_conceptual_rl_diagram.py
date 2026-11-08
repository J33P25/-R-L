"""
GreenMind: Conceptual RL Functionality Diagram (Section 3.6)
Draws the classic, iconic Reinforcement Learning Agent-Environment Conceptual Interaction Loop
customized for GreenMind autonomous building control.
Components explicitly illustrated:
1. Agent (Decision Maker: Policy & Value Function)
2. Environment (Building Physical System, Climate & Occupants)
3. State (S_t: Zone temperature, humidity, CO2, occupancy, weather)
4. Action (A_t: Chilled water valve, supply fan speed, fresh-air damper)
5. Reward (R_t: Energy savings, thermal comfort, air quality, actuator smoothness)
6. Policy Update Loop (Critic TD evaluation -> Policy gradient -> Updated policy)
7. Training vs. Inference Flow (Differentiating real-time execution vs. policy learning)
"""
import os 
import tempfile 
from pathlib import Path 
os .environ ['MPLCONFIGDIR']=tempfile .gettempdir ()
import matplotlib .pyplot as plt 
import matplotlib .patches as patches 
from matplotlib .lines import Line2D 
ROOT_DIR =Path (__file__ ).resolve ().parent .parent 
def draw_conceptual_rl_diagram ():
    fig ,ax =plt .subplots (figsize =(12 ,8.5 ),dpi =300 )
    ax .set_xlim (-0.5 ,12.5 )
    ax .set_ylim (-0.5 ,9.5 )
    ax .axis ('off')
    col_agent_bg ="#ecfdf5"
    col_agent_border ="#059669"
    col_env_bg ="#eff6ff"
    col_env_border ="#0284c7"
    col_inference ="#0284c7"
    col_training ="#dc2626"
    agent_patch =patches .FancyBboxPatch (
    (1.5 ,5.4 ),9.0 ,3.4 ,
    boxstyle ="round,pad=0.2,rounding_size=0.4",
    edgecolor =col_agent_border ,facecolor =col_agent_bg ,
    linewidth =2.5 ,zorder =2 
    )
    ax .add_patch (agent_patch )
    ax .text (6.0 ,8.35 ,"AGENT",fontsize =15 ,fontweight ='bold',ha ='center',color ="#065f46",zorder =3 )
    ax .text (6.0 ,7.95 ,"Autonomous Controller (Deep Actor-Critic / SAC)",fontsize =10 ,fontstyle ='italic',ha ='center',color ="#047857",zorder =3 )
    ax .plot ([3.5 ,8.5 ],[7.75 ,7.75 ],color ="#a7f3d0",lw =1.2 ,zorder =3 )
    actor_sub =patches .FancyBboxPatch (
    (2.0 ,5.8 ),3.6 ,1.6 ,
    boxstyle ="round,pad=0.1,rounding_size=0.2",
    edgecolor ="#0284c7",facecolor ="#ffffff",
    linewidth =1.4 ,zorder =3 
    )
    ax .add_patch (actor_sub )
    ax .text (3.8 ,6.8 ,r"Policy $\pi(a | s)$",fontsize =11 ,fontweight ='bold',ha ='center',color ="#0369a1",zorder =4 )
    ax .text (3.8 ,6.25 ,r"Inference Mode:"+"\n"+r"$s_t \mapsto a_t$ (Continuous)",fontsize =8.5 ,ha ='center',color ="#334155",zorder =4 )
    critic_sub =patches .FancyBboxPatch (
    (6.4 ,5.8 ),3.6 ,1.6 ,
    boxstyle ="round,pad=0.1,rounding_size=0.2",
    edgecolor =col_training ,facecolor ="#ffffff",
    linewidth =1.4 ,zorder =3 
    )
    ax .add_patch (critic_sub )
    ax .text (8.2 ,6.8 ,"Critic $Q(s, a)$ & Learner",fontsize =10.5 ,fontweight ='bold',ha ='center',color =col_training ,zorder =4 )
    ax .text (8.2 ,6.25 ,"Training Mode:\nBellman Loss & Policy Gradient",fontsize =8.5 ,ha ='center',color ="#334155",zorder =4 )
    ax .annotate (
    '',xy =(5.6 ,6.6 ),xytext =(6.4 ,6.6 ),
    arrowprops =dict (arrowstyle ="-|>",color =col_training ,lw =2.0 ,linestyle ="--",mutation_scale =14 ),
    zorder =5 
    )
    ax .text (6.0 ,7.0 ,r"Policy Update"+"\n"+r"$\Delta \theta$",fontsize =8 ,fontweight ='bold',ha ='center',color =col_training ,zorder =5 )
    env_patch =patches .FancyBboxPatch (
    (1.5 ,0.6 ),9.0 ,3.4 ,
    boxstyle ="round,pad=0.2,rounding_size=0.4",
    edgecolor =col_env_border ,facecolor =col_env_bg ,
    linewidth =2.5 ,zorder =2 
    )
    ax .add_patch (env_patch )
    ax .text (6.0 ,3.55 ,"ENVIRONMENT",fontsize =15 ,fontweight ='bold',ha ='center',color ="#0369a1",zorder =3 )
    ax .text (6.0 ,3.15 ,"Commercial Building System (Digital Twin)",fontsize =10 ,fontstyle ='italic',ha ='center',color ="#0284c7",zorder =3 )
    ax .plot ([3.5 ,8.5 ],[2.95 ,2.95 ],color ="#bae6fd",lw =1.2 ,zorder =3 )
    plant_sub =patches .FancyBboxPatch (
    (2.0 ,1.0 ),3.8 ,1.6 ,
    boxstyle ="round,pad=0.1,rounding_size=0.2",
    edgecolor ="#0284c7",facecolor ="#ffffff",
    linewidth =1.4 ,zorder =3 
    )
    ax .add_patch (plant_sub )
    ax .text (3.9 ,2.05 ,"Physical Dynamics",fontsize =10.5 ,fontweight ='bold',ha ='center',color ="#0284c7",zorder =4 )
    ax .text (3.9 ,1.45 ,r"• Thermal Mass (ISO 13790)"+"\n"+r"• $\mathrm{CO}_2$ Dilution (ASHRAE 62.1)"+"\n"+r"• Power Consumption (kW)",
    fontsize =8.0 ,ha ='center',color ="#334155",zorder =4 )
    dist_sub =patches .FancyBboxPatch (
    (6.2 ,1.0 ),3.8 ,1.6 ,
    boxstyle ="round,pad=0.1,rounding_size=0.2",
    edgecolor ="#64748b",facecolor ="#ffffff",
    linewidth =1.4 ,zorder =3 
    )
    ax .add_patch (dist_sub )
    ax .text (8.1 ,2.05 ,"Exogenous Factors",fontsize =10.5 ,fontweight ='bold',ha ='center',color ="#334155",zorder =4 )
    ax .text (8.1 ,1.45 ,r"• Outdoor Weather ($T_{\mathrm{out}}, RH_{\mathrm{out}}$)"+"\n"+r"• Solar Irradiance ($I_{\mathrm{solar}}$)"+"\n"+r"• Stochastic Occupancy ($N_{\mathrm{occ}}$)",
    fontsize =8.0 ,ha ='center',color ="#334155",zorder =4 )
    ax .annotate (
    '',xy =(9.2 ,4.0 ),xytext =(9.2 ,5.4 ),
    arrowprops =dict (arrowstyle ="-|>",color =col_inference ,lw =2.6 ,mutation_scale =18 ),
    zorder =5 
    )
    action_box =patches .FancyBboxPatch (
    (8.2 ,4.35 ),3.4 ,0.75 ,
    boxstyle ="round,pad=0.1,rounding_size=0.15",
    edgecolor =col_inference ,facecolor ="#ffffff",
    linewidth =1.4 ,zorder =6 
    )
    ax .add_patch (action_box )
    ax .text (9.9 ,4.85 ,r"Action $A_t$",fontsize =10 ,fontweight ='bold',ha ='center',color =col_inference ,zorder =7 )
    ax .text (9.9 ,4.5 ,r"Valve ($u_v$), Fan ($u_f$), Damper ($u_d$)",fontsize =7.5 ,ha ='center',color ="#334155",zorder =7 )
    ax .annotate (
    '',xy =(2.8 ,5.4 ),xytext =(2.8 ,4.0 ),
    arrowprops =dict (arrowstyle ="-|>",color =col_inference ,lw =2.6 ,mutation_scale =18 ),
    zorder =5 
    )
    state_box =patches .FancyBboxPatch (
    (0.5 ,4.35 ),3.4 ,0.75 ,
    boxstyle ="round,pad=0.1,rounding_size=0.15",
    edgecolor =col_inference ,facecolor ="#ffffff",
    linewidth =1.4 ,zorder =6 
    )
    ax .add_patch (state_box )
    ax .text (2.2 ,4.85 ,r"State $S_t$",fontsize =10 ,fontweight ='bold',ha ='center',color =col_inference ,zorder =7 )
    ax .text (2.2 ,4.5 ,r"$[T_{\mathrm{in}}, RH_{\mathrm{in}}, C_{\mathrm{in}}, N_{\mathrm{occ}}, \dots]$",fontsize =7.5 ,ha ='center',color ="#334155",zorder =7 )
    ax .annotate (
    '',xy =(6.0 ,5.4 ),xytext =(6.0 ,4.0 ),
    arrowprops =dict (arrowstyle ="-|>",color =col_training ,lw =2.4 ,linestyle ="--",mutation_scale =18 ),
    zorder =5 
    )
    reward_box =patches .FancyBboxPatch (
    (4.3 ,4.35 ),3.4 ,0.75 ,
    boxstyle ="round,pad=0.1,rounding_size=0.15",
    edgecolor =col_training ,facecolor ="#ffffff",
    linewidth =1.4 ,zorder =6 
    )
    ax .add_patch (reward_box )
    ax .text (6.0 ,4.85 ,r"Reward $R_t$",fontsize =10 ,fontweight ='bold',ha ='center',color =col_training ,zorder =7 )
    ax .text (6.0 ,4.5 ,r"Energy + Comfort + IAQ + Smoothness",fontsize =7.5 ,ha ='center',color ="#334155",zorder =7 )
    ax .text (6.0 ,0.15 ,r"Discrete Decision Epoch: $\Delta t = 15$ Minutes ($96$ Time Steps per Diurnal Day)",
    fontsize =9.0 ,fontstyle ='italic',ha ='center',color ="#64748b",zorder =3 )
    legend_elements =[
    Line2D ([0 ],[0 ],color =col_inference ,lw =2.6 ,label =r'Online Inference Flow: Real-Time Action Execution ($S_t \mapsto A_t$)'),
    Line2D ([0 ],[0 ],color =col_training ,lw =2.2 ,linestyle ='--',label =r'Training / Policy Update Loop: Learning from Feedback ($R_t \mapsto \Delta \theta, \Delta \psi$)')
    ]
    leg =ax .legend (handles =legend_elements ,loc ='upper center',bbox_to_anchor =(0.5 ,0.0 ),
    ncol =2 ,frameon =True ,facecolor ='#ffffff',edgecolor ='#cbd5e1',fontsize =9.0 ,framealpha =1.0 )
    leg .get_frame ().set_linewidth (1.0 )
    plt .tight_layout ()
    p1 =ROOT_DIR /"rl_conceptual_functionality_diagram.png"
    p2 =ROOT_DIR /"logs"/"rl_conceptual_functionality_diagram.png"
    plt .savefig (p1 ,dpi =300 ,bbox_inches ='tight')
    plt .savefig (p2 ,dpi =300 ,bbox_inches ='tight')
    plt .close ()
    print (f"[+] Saved {p1 .name } and {p2 .name } successfully.")
if __name__ =='__main__':
    draw_conceptual_rl_diagram ()
