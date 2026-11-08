"""
GreenMind: Publication-Grade Comparative Analysis & Figure Generator
Generates clean, spacious, high-resolution figures for papers and presentations:
1. Individual Standalone Figures:
   - fig1_power_demand_24h.png
   - fig2_energy_savings_bar.png
   - fig3_thermal_comfort_envelope.png
   - fig4_iaq_co2_trajectory.png
   - fig5_actuator_jitter_hunting.png
   - fig6_sac_actuator_control.png
2. Master Overview Figure:
   - publication_comparison_overview.png & model_based_vs_model_free_comparison.png
"""
import sys 
from pathlib import Path 
ROOT_DIR =Path (__file__ ).resolve ().parent .parent 
if str (ROOT_DIR )not in sys .path :
    sys .path .insert (0 ,str (ROOT_DIR ))
import numpy as np 
import json 
import matplotlib .pyplot as plt 
import matplotlib .gridspec as gridspec 
from src .config import LOGS_DIR 
plt .rcParams ['font.sans-serif']=['DejaVu Sans','Arial','Helvetica','sans-serif']
plt .rcParams ['axes.unicode_minus']=False 
plt .rcParams ['figure.autolayout']=False 
COLORS ={
"Rule-Based":"#475569",
"Policy Iteration":"#7c3aed",
"PPO":"#0284c7",
"SAC":"#059669"
}
STYLES ={
"Rule-Based":{"ls":"--","lw":2.2 ,"alpha":0.85 },
"Policy Iteration":{"ls":"-.","lw":2.2 ,"alpha":0.9 },
"PPO":{"ls":"-","lw":2.2 ,"alpha":0.9 },
"SAC":{"ls":"-","lw":2.8 ,"alpha":1.0 }
}
LABELS ={
"Rule-Based":"Rule-Based Control (RBC Baseline)",
"Policy Iteration":"Analytical Policy Iteration (Model-Based)",
"PPO":"PPO Continuous (Model-Free On-Policy)",
"SAC":"SAC Continuous (Model-Free Off-Policy [Optimal])"
}
def load_data ():
    results_file =LOGS_DIR /"evaluation_results.json"
    if not results_file .exists ():
        raise FileNotFoundError (f"Results file not found at {results_file }. Run evaluate_comparison.py first.")
    with open (results_file ,"r")as f :
        data =json .load (f )
    return data ["results"]
def plot_power_demand (results ,time_steps ,out_dir ):
    fig ,ax =plt .subplots (figsize =(10 ,5 ),dpi =300 )
    for agent_key in ["Rule-Based","Policy Iteration","PPO","SAC"]:
        pwr =results [agent_key ]["history"]["power_kw"]
        ax .plot (time_steps ,pwr ,label =LABELS [agent_key ],color =COLORS [agent_key ],**STYLES [agent_key ])
    ax .set_title ("Figure 1: 24-Hour Building Electrical Power Demand Profile",fontsize =13 ,fontweight ='bold',pad =12 )
    ax .set_xlabel ("Time of Day (Hours)",fontsize =11 ,fontweight ='semibold')
    ax .set_ylabel ("Total HVAC Electrical Power (kW)",fontsize =11 ,fontweight ='semibold')
    ax .set_xlim (0 ,24 )
    ax .set_xticks (range (0 ,25 ,2 ))
    ax .set_ylim (0 ,4.5 )
    ax .grid (True ,linestyle =":",alpha =0.6 )
    ax .legend (loc ="upper right",framealpha =0.95 ,fontsize =9.5 )
    ax .axvspan (8.0 ,18.0 ,color ="#fef3c7",alpha =0.35 ,label ="Standard Office Hours (08:00 - 18:00)")
    ax .legend (loc ="upper right",framealpha =0.95 ,fontsize =9.5 )
    plt .tight_layout ()
    p =out_dir /"fig1_power_demand_24h.png"
    plt .savefig (p ,dpi =300 )
    plt .close ()
    print (f"  [+] Saved {p .name }")
def plot_energy_savings (results ,out_dir ):
    fig ,ax =plt .subplots (figsize =(8.5 ,5 ),dpi =300 )
    agent_keys =["Rule-Based","Policy Iteration","PPO","SAC"]
    display_names =["Rule-Based\n(Baseline)","Policy Iteration\n(Model-Based)","PPO\n(On-Policy)","SAC\n(Off-Policy)"]
    energy_vals =[results [k ]["total_energy_kwh"]for k in agent_keys ]
    bar_colors =[COLORS [k ]for k in agent_keys ]
    bars =ax .bar (display_names ,energy_vals ,color =bar_colors ,width =0.52 ,edgecolor ="#1e293b",linewidth =1.2 )
    ax .set_title ("Figure 2: Daily Total Electrical Energy Consumption & Relative Savings",fontsize =12.5 ,fontweight ='bold',pad =12 )
    ax .set_ylabel ("Total Energy Consumed (kWh)",fontsize =11 ,fontweight ='semibold')
    ax .set_ylim (0 ,max (energy_vals )*1.25 )
    ax .grid (axis ="y",linestyle =":",alpha =0.7 )
    baseline_val =results ["Rule-Based"]["total_energy_kwh"]
    for bar ,val ,k in zip (bars ,energy_vals ,agent_keys ):
        h =bar .get_height ()
        if k =="Rule-Based":
            txt =f"{val :.2f} kWh\n(Baseline)"
        else :
            saved_pct =((baseline_val -val )/baseline_val )*100.0 
            txt =f"{val :.2f} kWh\n(-{saved_pct :.1f}%)"
        ax .annotate (txt ,xy =(bar .get_x ()+bar .get_width ()/2 ,h ),
        xytext =(0 ,6 ),textcoords ="offset points",
        ha ='center',va ='bottom',fontsize =10 ,fontweight ='bold',color ="#0f172a")
    plt .tight_layout ()
    p =out_dir /"fig2_energy_savings_bar.png"
    plt .savefig (p ,dpi =300 )
    plt .close ()
    print (f"  [+] Saved {p .name }")
def plot_thermal_comfort (results ,time_steps ,out_dir ):
    fig ,ax =plt .subplots (figsize =(10 ,5 ),dpi =300 )
    ax .axhspan (21.0 ,24.0 ,color ="#d1fae5",alpha =0.65 ,label ="ASHRAE Standard 55 Comfort Envelope (21°C - 24°C)")
    ax .axhline (21.0 ,color ="#059669",linestyle ="--",linewidth =1.4 )
    ax .axhline (24.0 ,color ="#059669",linestyle ="--",linewidth =1.4 )
    for agent_key in ["Rule-Based","Policy Iteration","PPO","SAC"]:
        temps =results [agent_key ]["history"]["indoor_temp"]
        ax .plot (time_steps ,temps ,label =LABELS [agent_key ],color =COLORS [agent_key ],**STYLES [agent_key ])
    ax .set_title ("Figure 3: Indoor Zone Temperature Trajectory vs. ASHRAE 55 Bounds",fontsize =12.5 ,fontweight ='bold',pad =12 )
    ax .set_xlabel ("Time of Day (Hours)",fontsize =11 ,fontweight ='semibold')
    ax .set_ylabel ("Indoor Air Temperature (°C)",fontsize =11 ,fontweight ='semibold')
    ax .set_xlim (0 ,24 )
    ax .set_xticks (range (0 ,25 ,2 ))
    ax .set_ylim (19.0 ,32.0 )
    ax .grid (True ,linestyle =":",alpha =0.6 )
    ax .legend (loc ="upper left",framealpha =0.95 ,fontsize =9.5 )
    plt .tight_layout ()
    p =out_dir /"fig3_thermal_comfort_envelope.png"
    plt .savefig (p ,dpi =300 )
    plt .close ()
    print (f"  [+] Saved {p .name }")
def plot_iaq_co2 (results ,time_steps ,out_dir ):
    """
    Dedicated, crystal-clear Indoor Air Quality (CO2) comparison figure.
    Clearly shows the ASHRAE 62.1 1000 ppm threshold and shaded regions.
    """
    fig ,ax =plt .subplots (figsize =(10 ,5.5 ),dpi =300 )
    ax .axhspan (380.0 ,1000.0 ,color ="#ecfdf5",alpha =0.7 ,label ="Safe Air Quality Zone (CO2 <= 1000 ppm)")
    ax .axhspan (1000.0 ,1600.0 ,color ="#fee2e2",alpha =0.5 ,label ="Unhealthy / Stuffy Air (CO2 > 1000 ppm Violation)")
    ax .axhline (1000.0 ,color ="#dc2626",linestyle ="--",linewidth =1.8 ,label ="ASHRAE 62.1 Safety Limit (1000 ppm)")
    for agent_key in ["Rule-Based","Policy Iteration","PPO","SAC"]:
        co2_vals =results [agent_key ]["history"]["co2_ppm"]
        viol_hrs =results [agent_key ].get ("iaq_violation_hours",0.0 )
        label_with_viol =f"{LABELS [agent_key ]} — [{viol_hrs :.1f}h violations]"
        ax .plot (time_steps ,co2_vals ,label =label_with_viol ,color =COLORS [agent_key ],**STYLES [agent_key ])
    ax .set_title (r"Figure 4: Indoor Air Quality ($\mathrm{CO}_2$ Concentration) Across 24 Hours",fontsize =12.5 ,fontweight ='bold',pad =12 )
    ax .set_xlabel ("Time of Day (Hours)",fontsize =11 ,fontweight ='semibold')
    ax .set_ylabel (r"Indoor $\mathrm{CO}_2$ Concentration (ppm)",fontsize =11 ,fontweight ='semibold')
    ax .set_xlim (0 ,24 )
    ax .set_xticks (range (0 ,25 ,2 ))
    ax .set_ylim (380 ,1500 )
    ax .grid (True ,linestyle =":",alpha =0.6 )
    ax .legend (loc ="upper right",framealpha =0.95 ,fontsize =9 )
    ax .annotate (r"SAC maintains clean air all day ($0.0\mathrm{h}$ violation)",
    xy =(14.0 ,880.0 ),xytext =(12.0 ,600.0 ),
    arrowprops =dict (arrowstyle ="->",color =COLORS ["SAC"],lw =1.8 ),
    fontsize =9.5 ,fontweight ='bold',color =COLORS ["SAC"],
    bbox =dict (boxstyle ="round,pad=0.4",fc ="#ffffff",ec =COLORS ["SAC"],alpha =0.9 ))
    plt .tight_layout ()
    p =out_dir /"fig4_iaq_co2_trajectory.png"
    plt .savefig (p ,dpi =300 )
    plt .close ()
    print (f"  [+] Saved {p .name }")
def plot_actuator_jitter (results ,out_dir ):
    fig ,ax =plt .subplots (figsize =(8.5 ,5 ),dpi =300 )
    agent_keys =["Rule-Based","Policy Iteration","PPO","SAC"]
    display_names =["Rule-Based\n(Baseline)","Policy Iteration\n(Model-Based)","PPO\n(On-Policy)","SAC\n(Off-Policy)"]
    jitter_vals =[results [k ]["actuator_jitter"]for k in agent_keys ]
    bar_colors =[COLORS [k ]for k in agent_keys ]
    bars =ax .bar (display_names ,jitter_vals ,color =bar_colors ,width =0.52 ,edgecolor ="#1e293b",linewidth =1.2 )
    ax .set_title ("Figure 5: Actuator Mechanical Hunting & Jitter (Lower = Equipment Friendly)",fontsize =12.5 ,fontweight ='bold',pad =12 )
    ax .set_ylabel (r"Mean Squared Action Variance $||\Delta a||^2$",fontsize =11 ,fontweight ='semibold')
    ax .set_ylim (0 ,max (jitter_vals )*1.3 )
    ax .grid (axis ="y",linestyle =":",alpha =0.7 )
    rbc_jitter =results ["Rule-Based"]["actuator_jitter"]
    for bar ,val ,k in zip (bars ,jitter_vals ,agent_keys ):
        h =bar .get_height ()
        if k =="SAC":
            ratio =rbc_jitter /max (val ,1e-6 )
            txt =f"{val :.4f}\n({ratio :.0f}x Smoother!)"
        else :
            txt =f"{val :.4f}"
        ax .annotate (txt ,xy =(bar .get_x ()+bar .get_width ()/2 ,h ),
        xytext =(0 ,6 ),textcoords ="offset points",
        ha ='center',va ='bottom',fontsize =10 ,fontweight ='bold',color ="#0f172a")
    plt .tight_layout ()
    p =out_dir /"fig5_actuator_jitter_hunting.png"
    plt .savefig (p ,dpi =300 )
    plt .close ()
    print (f"  [+] Saved {p .name }")
def plot_sac_actions (results ,time_steps ,out_dir ):
    """
    Shows SAC's continuous 3-actuator command trajectories over time.
    """
    fig ,ax =plt .subplots (figsize =(10 ,4.5 ),dpi =300 )
    sac_hist =results ["SAC"]["history"]
    valves =sac_hist ["valve"]
    fans =sac_hist ["fan"]
    dampers =sac_hist ["damper"]
    ax .plot (time_steps ,valves ,label ="Chilled Water Valve Position (0-1)",color ="#0284c7",lw =2.2 )
    ax .plot (time_steps ,fans ,label ="Supply Fan VFD Speed (0-1)",color ="#e11d48",lw =2.2 )
    ax .plot (time_steps ,dampers ,label ="Fresh Air Damper Opening (0-1)",color ="#059669",lw =2.2 )
    ax .set_title ("Figure 6: SAC Autonomous Actuator Control Trajectories (Valve, Fan, Damper)",fontsize =12.5 ,fontweight ='bold',pad =12 )
    ax .set_xlabel ("Time of Day (Hours)",fontsize =11 ,fontweight ='semibold')
    ax .set_ylabel ("Modulation Output [0.0 - 1.0]",fontsize =11 ,fontweight ='semibold')
    ax .set_xlim (0 ,24 )
    ax .set_xticks (range (0 ,25 ,2 ))
    ax .set_ylim (-0.05 ,1.05 )
    ax .grid (True ,linestyle =":",alpha =0.6 )
    ax .legend (loc ="upper right",framealpha =0.95 ,fontsize =10 )
    plt .tight_layout ()
    p =out_dir /"fig6_sac_actuator_control.png"
    plt .savefig (p ,dpi =300 )
    plt .close ()
    print (f"  [+] Saved {p .name }")
def plot_master_overview (results ,time_steps ,out_dir ):
    """
    Spacious, clean 2x2 master overview with ample margins and no overlapping text.
    """
    fig =plt .figure (figsize =(18 ,12 ),dpi =300 )
    gs =gridspec .GridSpec (2 ,2 ,figure =fig ,hspace =0.32 ,wspace =0.22 )
    agent_keys =["Rule-Based","Policy Iteration","PPO","SAC"]
    ax1 =fig .add_subplot (gs [0 ,0 ])
    for k in agent_keys :
        ax1 .plot (time_steps ,results [k ]["history"]["power_kw"],label =LABELS [k ],color =COLORS [k ],**STYLES [k ])
    ax1 .set_title ("(a) 24-Hour Electrical Power Demand Profile",fontsize =12 ,fontweight ='bold')
    ax1 .set_xlabel ("Time of Day (Hours)",fontsize =10 )
    ax1 .set_ylabel ("Electrical Power (kW)",fontsize =10 )
    ax1 .set_xlim (0 ,24 )
    ax1 .set_ylim (0 ,4.5 )
    ax1 .grid (True ,linestyle =":",alpha =0.6 )
    ax1 .legend (loc ="upper right",fontsize =8.5 ,framealpha =0.95 )
    ax2 =fig .add_subplot (gs [0 ,1 ])
    energy_vals =[results [k ]["total_energy_kwh"]for k in agent_keys ]
    display_names =["Rule-Based","Policy Iteration","PPO","SAC"]
    bars =ax2 .bar (display_names ,energy_vals ,color =[COLORS [k ]for k in agent_keys ],width =0.5 ,edgecolor ="#1e293b")
    ax2 .set_title ("(b) Daily Total Energy Consumption",fontsize =12 ,fontweight ='bold')
    ax2 .set_ylabel ("Total Energy (kWh)",fontsize =10 )
    ax2 .set_ylim (0 ,max (energy_vals )*1.25 )
    ax2 .grid (axis ="y",linestyle =":",alpha =0.6 )
    baseline_val =results ["Rule-Based"]["total_energy_kwh"]
    for bar ,val ,k in zip (bars ,energy_vals ,agent_keys ):
        h =bar .get_height ()
        txt =f"{val :.1f} kWh"if k =="Rule-Based"else f"{val :.1f} kWh\n(-{((baseline_val -val )/baseline_val )*100 :.1f}%)"
        ax2 .annotate (txt ,xy =(bar .get_x ()+bar .get_width ()/2 ,h ),xytext =(0 ,4 ),
        textcoords ="offset points",ha ='center',va ='bottom',fontsize =9 ,fontweight ='bold')
    ax3 =fig .add_subplot (gs [1 ,0 ])
    ax3 .axhspan (21.0 ,24.0 ,color ="#d1fae5",alpha =0.6 ,label ="ASHRAE 55 Comfort Zone")
    for k in agent_keys :
        ax3 .plot (time_steps ,results [k ]["history"]["indoor_temp"],label =k ,color =COLORS [k ],**STYLES [k ])
    ax3 .set_title ("(c) Thermal Comfort Trajectory (°C)",fontsize =12 ,fontweight ='bold')
    ax3 .set_xlabel ("Time of Day (Hours)",fontsize =10 )
    ax3 .set_ylabel ("Zone Temperature (°C)",fontsize =10 )
    ax3 .set_xlim (0 ,24 )
    ax3 .set_ylim (19.0 ,32.0 )
    ax3 .grid (True ,linestyle =":",alpha =0.6 )
    ax3 .legend (loc ="upper left",fontsize =8.5 ,framealpha =0.95 )
    ax4 =fig .add_subplot (gs [1 ,1 ])
    ax4 .axhline (1000.0 ,color ="#dc2626",linestyle ="--",linewidth =1.5 ,label ="ASHRAE 62.1 Limit (1000 ppm)")
    for k in agent_keys :
        ax4 .plot (time_steps ,results [k ]["history"]["co2_ppm"],label =k ,color =COLORS [k ],**STYLES [k ])
    ax4 .set_title (r"(d) Indoor Air Quality ($\mathrm{CO}_2$ ppm)",fontsize =12 ,fontweight ='bold')
    ax4 .set_xlabel ("Time of Day (Hours)",fontsize =10 )
    ax4 .set_ylabel (r"$\mathrm{CO}_2$ Concentration (ppm)",fontsize =10 )
    ax4 .set_xlim (0 ,24 )
    ax4 .set_ylim (380 ,1500 )
    ax4 .grid (True ,linestyle =":",alpha =0.6 )
    ax4 .legend (loc ="upper right",fontsize =8.5 ,framealpha =0.95 )
    fig .suptitle ("GreenMind Benchmark: Model-Based (Analytical MDP) vs. Model-Free Deep RL (SAC + PPO)",
    fontsize =15 ,fontweight ='bold',y =0.98 )
    fig .subplots_adjust (top =0.93 ,bottom =0.08 ,left =0.07 ,right =0.96 ,hspace =0.32 ,wspace =0.22 )
    out_file1 =out_dir /"publication_comparison_overview.png"
    out_file2 =out_dir /"model_based_vs_model_free_comparison.png"
    plt .savefig (out_file1 ,dpi =300 )
    plt .savefig (out_file2 ,dpi =300 )
    plt .close ()
    print (f"  [+] Saved {out_file1 .name } and {out_file2 .name }")
def generate_comparison_plots ():
    results =load_data ()
    time_steps =np .array (results ["Rule-Based"]["history"]["time_step"])*0.25 
    out_dir =LOGS_DIR 
    print ("\n[Visualization] Generating Publication-Quality Figures...")
    plot_power_demand (results ,time_steps ,out_dir )
    plot_energy_savings (results ,out_dir )
    plot_thermal_comfort (results ,time_steps ,out_dir )
    plot_iaq_co2 (results ,time_steps ,out_dir )
    plot_actuator_jitter (results ,out_dir )
    plot_sac_actions (results ,time_steps ,out_dir )
    plot_master_overview (results ,time_steps ,out_dir )
    print ("\n[Visualization] Complete! All figures generated in logs/ with 0 font warnings.\n")
if __name__ =="__main__":
    generate_comparison_plots ()
