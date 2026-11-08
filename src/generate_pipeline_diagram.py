import matplotlib .pyplot as plt 
import matplotlib .patches as patches 
def draw_pipeline ():
    fig ,ax =plt .subplots (figsize =(15 ,2.6 ),dpi =300 )
    ax .set_xlim (-0.2 ,14.8 )
    ax .set_ylim (-0.4 ,2.2 )
    ax .axis ('off')
    box_edge ='#2c3e50'
    box_bg ='#fdfefe'
    text_main ='#1a252f'
    text_sub ='#566573'
    arrow_color ='#2c3e50'
    feedback_color ='#7f8c8d'
    boxes =[
    ("Environment","(Building\nThermal & CO2)",0.0 ,0.4 ,1.45 ,1.1 ),
    ("State Extraction","(14-D State\nVector $s_t$)",1.85 ,0.4 ,1.55 ,1.1 ),
    ("Agent","(PPO / SAC)",3.8 ,0.4 ,1.4 ,1.1 ),
    ("Policy / Value\nNetwork","(Actor $\\pi_\\theta$ &\nCritic $V_\\phi$)",5.6 ,0.4 ,1.6 ,1.1 ),
    ("Action","(Continuous\n$a_t = [v, f]$)",7.6 ,0.4 ,1.5 ,1.1 ),
    ("Environment\nUpdate","(Physics Step\n$\\Delta t = 15$m)",9.5 ,0.4 ,1.55 ,1.1 ),
    ("Reward","(Multi-Objective\n+ Wear)",11.45 ,0.4 ,1.5 ,1.1 ),
    ("Training Loop","(Loss &\nBackprop)",13.35 ,0.4 ,1.4 ,1.1 ),
    ]
    box_coords ={}
    for title ,subtitle ,x ,y ,w ,h in boxes :
        rect =patches .FancyBboxPatch (
        (x ,y ),w ,h ,
        boxstyle ="round,pad=0.04,rounding_size=0.08",
        edgecolor =box_edge ,facecolor =box_bg ,
        linewidth =1.4 ,zorder =2 
        )
        ax .add_patch (rect )
        mid_x =x +w /2.0 
        mid_y =y +h /2.0 
        ax .text (mid_x ,mid_y +0.16 ,title ,ha ='center',va ='center',
        fontsize =9.5 ,fontweight ='bold',color =text_main ,zorder =3 )
        ax .text (mid_x ,mid_y -0.22 ,subtitle ,ha ='center',va ='center',
        fontsize =8.0 ,fontstyle ='italic',color =text_sub ,zorder =3 )
        box_coords [title .split ('\n')[0 ]]=(x ,y ,w ,h )
    for i in range (len (boxes )-1 ):
        x1 =boxes [i ][2 ]+boxes [i ][4 ]
        y1 =boxes [i ][3 ]+boxes [i ][5 ]/2.0 
        x2 =boxes [i +1 ][2 ]
        y2 =boxes [i +1 ][3 ]+boxes [i +1 ][5 ]/2.0 
        ax .annotate (
        '',xy =(x2 ,y2 ),xytext =(x1 ,y1 ),
        arrowprops =dict (arrowstyle ="-|>",color =arrow_color ,lw =1.5 ,mutation_scale =12 ),
        zorder =4 
        )
    t_x =13.35 +1.4 /2.0 
    t_y =0.4 +1.1 
    p_x =5.6 +1.6 /2.0 
    p_y =0.4 +1.1 
    y_top =1.95 
    ax .plot ([t_x ,t_x ,p_x ,p_x ],[t_y ,y_top ,y_top ,p_y +0.05 ],
    linestyle =':',color =feedback_color ,linewidth =1.5 ,zorder =1 )
    ax .annotate (
    '',xy =(p_x ,p_y ),xytext =(p_x ,p_y +0.08 ),
    arrowprops =dict (arrowstyle ="-|>",color =feedback_color ,lw =1.5 ,mutation_scale =12 ),
    zorder =4 
    )
    ax .text ((t_x +p_x )/2.0 ,y_top ,"  Update Weights (Backprop)  ",
    ha ='center',va ='center',fontsize =8.0 ,fontweight ='bold',color ='#2c3e50',
    bbox =dict (boxstyle ='round,pad=0.2',facecolor ='#e5e8e8',edgecolor ='#bdc3c7',lw =0.8 ),
    zorder =5 )
    u_x =9.5 +1.55 /2.0 
    u_y =0.4 
    e_x =0.0 +1.45 /2.0 
    e_y =0.4 
    y_bot =-0.15 
    ax .plot ([u_x ,u_x ,e_x ,e_x ],[u_y ,y_bot ,y_bot ,e_y -0.05 ],
    linestyle =':',color =feedback_color ,linewidth =1.5 ,zorder =1 )
    ax .annotate (
    '',xy =(e_x ,e_y ),xytext =(e_x ,e_y -0.08 ),
    arrowprops =dict (arrowstyle ="-|>",color =feedback_color ,lw =1.5 ,mutation_scale =12 ),
    zorder =4 
    )
    ax .text ((u_x +e_x )/2.0 ,y_bot ,"  Next State $s_{t+1}$  ",
    ha ='center',va ='center',fontsize =8.0 ,fontweight ='bold',color ='#2c3e50',
    bbox =dict (boxstyle ='round,pad=0.2',facecolor ='#e5e8e8',edgecolor ='#bdc3c7',lw =0.8 ),
    zorder =5 )
    plt .tight_layout ()
    plt .savefig ('rl_pipeline_block_diagram.png',dpi =300 ,bbox_inches ='tight',transparent =False )
    plt .close ()
    print ("Generated rl_pipeline_block_diagram.png successfully.")
if __name__ =='__main__':
    draw_pipeline ()
