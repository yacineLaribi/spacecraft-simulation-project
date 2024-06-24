from django.conf import settings
from django.shortcuts import render
import scipy.io
import pandas as pd
import os
import json

def index(request):
    return render(request, 'index.html')


def about(request):
    return render(request, "about.html")


import base64
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from django.shortcuts import render

def load_data():
    data = {}
    data['teta_pd_1x'] = scipy.io.loadmat('Pd_control/teta_pd_1x.mat')['teta_pd_1x']
    data['teta_p_pd_1x'] = scipy.io.loadmat('Pd_control/teta_p_pd_1x.mat')['teta_p_pd_1x']
    data['w_pd_1x'] = scipy.io.loadmat('Pd_control/w_pd_1x.mat')['w_pd_1x']
    data['d0_pd_1x'] = scipy.io.loadmat('Pd_control/d0_pd_1x.mat')['d0_pd_1x']
    data['t_pd_1x'] = scipy.io.loadmat('Pd_control/t_pd_1x.mat')['t_pd_1x']
    data['teta_do_1x'] = scipy.io.loadmat('Dobc_control/teta_do_1x.mat')['teta_do_1x']
    data['teta_p_do_1x'] = scipy.io.loadmat('Dobc_control/teta_p_do_1x.mat')['teta_p_do_1x']
    data['w_do_1x'] = scipy.io.loadmat('Dobc_control/w_do_1x.mat')['w_do_1x']
    data['d0_do_1x'] = scipy.io.loadmat('Dobc_control/d0_do_1x.mat')['d0_do_1x']
    data['t_do_1x'] = scipy.io.loadmat('Dobc_control/t_do_1x.mat')['t_do_1x']
    data['teta_es_1x'] = scipy.io.loadmat('Eso_control/teta_es_1x.mat')['teta_es_1x']
    data['teta_p_es_1x'] = scipy.io.loadmat('Eso_control/teta_p_es_1x.mat')['teta_p_es_1x']
    data['w_es_1x'] = scipy.io.loadmat('Eso_control/w_es_1x.mat')['w_es_1x']
    data['d0_es_1x'] = scipy.io.loadmat('Eso_control/d0_es_1x.mat')['d0_es_1x']
    data['t_es_1x'] = scipy.io.loadmat('Eso_control/t_es_1x.mat')['t_es_1x']

    # load the case 01 data #
    data['teta_pd_3x'] = scipy.io.loadmat('Pd_Dobc_control/teta_dobc_pd_3x.mat')['teta_dobc_pd_3x']
    data['teta_p_pd_3x'] = scipy.io.loadmat('Pd_Dobc_control/teta_p_dobc_pd_3x.mat')['teta_p_dobc_pd_3x']
    data['w_pd_3x'] = scipy.io.loadmat('Pd_Dobc_control/w_dobc_pd_3x.mat')['w_dobc_pd_3x']
    data['d_pd_3x'] = scipy.io.loadmat('Pd_Dobc_control/d_dobc_pd_3x.mat')['d_dobc_pd_3x']
    data['t_pd_3x'] = scipy.io.loadmat('Pd_Dobc_control/t_dobc_pd_3x.mat')['t_dobc_pd_3x']
    #rms_pd=scipy.io.loadmat('Pd_Dobc_control/RMS_Dobc_pd.mat')
    #----------------------------------------------------------------------------------------------------------------------------------------------------------#
    # load the case 02 data #
    # config sign
    data['teta_sm_sgn_3x'] = scipy.io.loadmat('SM_control/sign/teta_SM_sgn_3x.mat')['teta_SM_sgn_3x']
    data['teta_p_sm_sgn_3x'] = scipy.io.loadmat('SM_control/sign/teta_p_SM_sgn_3x.mat')['teta_p_SM_sgn_3x']
    # config sat 
    data['teta_sm_sat_3x'] = scipy.io.loadmat('SM_control/sat/teta_SM_sat_3x.mat')['teta_SM_sat_3x']
    data['teta_p_sm_sat_3x'] = scipy.io.loadmat('SM_control/sat/teta_p_SM_sat_3x.mat')['teta_p_SM_sat_3x']
    # config tanh
    data['teta_sm_tanh_3x'] = scipy.io.loadmat('SM_control/tanh/teta_SM_tanh_3x.mat')['teta_SM_tanh_3x']
    data['teta_p_sm_tanh_3x'] = scipy.io.loadmat('SM_control/tanh/teta_p_SM_tanh_3x.mat')['teta_p_SM_tanh_3x']
    # general 
    #rms_sm=scipy.io.loadmat('SM_control/RMS_SM.mat')
    data['t_sm_3x'] = scipy.io.loadmat('SM_control/t_SM_3x.mat')['t_SM_3x']
    data['w_sm_3x'] = scipy.io.loadmat('SM_control/w_SM_3x.mat')['w_SM_3x']
    #----------------------------------------------------------------------------------------------------------------------------------------------------------#
    # load the case 03 data #
    # first ASM
    data['teta_sm1_3x'] = scipy.io.loadmat('ASM_control/ASM1_control/teta_ASM1_3x.mat')['teta_ASM1_3x']
    data['teta_p_sm1_3x'] = scipy.io.loadmat('ASM_control/ASM1_control/teta_p_ASM1_3x.mat')['teta_p_ASM1_3x']
    data['t_sm1_3x'] = scipy.io.loadmat('ASM_control/ASM1_control/t_ASM1_3x.mat')['t_ASM1_3x']
    #rms_asm1=scipy.io.loadmat('ASM_control/ASM1_control/RMS_ASM1_3x.mat')
    # second ASM 
    data['teta_sm2_3x'] = scipy.io.loadmat('ASM_control/ASM2_control/teta_ASM2_3x.mat')['teta_ASM2_3x']
    data['teta_p_sm2_3x'] = scipy.io.loadmat('ASM_control/ASM2_control/teta_p_ASM2_3x.mat')['teta_p_ASM2_3x']
    data['t_sm2_3x'] = scipy.io.loadmat('ASM_control/ASM2_control/t_ASM2_3x.mat')['t_ASM2_3x']
    #rms_asm2=scipy.io.loadmat('ASM_control/ASM1_control/RMS_ASM2_3x.mat')
    #----------------------------------------------------------------------------------------------------------------------------------------------------------#

    # adjust the case 01 data #
    # value raw !!
    # data['teta_pd_3x'] = data['teta_pd_3x']["teta_dobc_pd_3x"]
    # data['teta_p_pd_3x'] = data['teta_p_pd_3x']["teta_p_dobc_pd_3x"]
    # data['t_pd_3x'] = data['t_pd_3x']["t_dobc_pd_3x"]
    # data['d_pd_3x'] = data['d_pd_3x']["d_dobc_pd_3x"]
    # data['w_pd_3x'] = data['w_pd_3x']["w_dobc_pd_3x"]
    # #----------------------------------------------------------------------------------------------------------------------------------------------------------#
    # # adjust the case 02 data #
    # # value raw !!
    # # config sign
    # data['teta_sm_sgn_3x'] = data['teta_sm_sgn_3x']["teta_SM_sgn_3x"]
    # data['teta_p_sm_sgn_3x'] = data['teta_p_sm_sgn_3x']["teta_p_SM_sgn_3x"]
    # # config sat 
    # data['teta_sm_sat_3x'] = data['teta_sm_sat_3x']["teta_SM_sat_3x"]
    # data['teta_p_sm_sat_3x'] = data['teta_p_sm_sat_3x']["teta_p_SM_sat_3x"]
    # # config tanh
    # data['teta_sm_tanh_3x'] = data['teta_sm_tanh_3x']["teta_SM_tanh_3x"]
    # data['teta_p_sm_tanh_3x'] = data['teta_p_sm_tanh_3x']["teta_p_SM_tanh_3x"]
    # # general
    # data['t_sm_3x'] = data['t_sm_3x']["t_SM_3x"]
    # data['w_sm_3x'] = data['w_sm_3x']["w_SM_3x"]
    # #----------------------------------------------------------------------------------------------------------------------------------------------------------#
    # # adjust the case 03 data #
    # # value raw !!
    # # ASM1
    # data['teta_sm1_3x'] = data['teta_sm1_3x']["teta_ASM1_3x"]
    # data['teta_p_sm1_3x'] = data['teta_p_sm1_3x']["teta_p_ASM1_3x"]
    # data['t_sm1_3x'] = data['t_sm1_3x']["t_ASM1_3x"]
    # # ASM2
    # data['teta_sm2_3x'] = data['teta_sm2_3x']["teta_ASM2_3x"]
    # data['teta_p_sm2_3x'] = data['teta_p_sm2_3x']["teta_p_ASM2_3x"]
    # data['t_sm2_3x'] = data['t_sm2_3x']["t_ASM2_3x"]
    #----------------------------------------------------------------------------------------------------------------------------------------------------------#

    # to datafram extention data 01#
    data['teta_pd_3x'] = pd.DataFrame(data['teta_pd_3x'])
    data['teta_p_pd_3x'] = pd.DataFrame(data['teta_p_pd_3x'])
    data['w_pd_3x'] = pd.DataFrame(data['w_pd_3x'])
    data['d_pd_3x'] = pd.DataFrame(data['d_pd_3x'])
    data['t_pd_3x'] = pd.DataFrame(data['t_pd_3x'])
    #----------------------------------------------------------------------------------------------------------------------------------------------------------#
    # to datafram extention data 02#
    # sign
    data['teta_sm_sgn_3x'] = pd.DataFrame(data['teta_sm_sgn_3x'])
    data['teta_p_sm_sgn_3x'] = pd.DataFrame(data['teta_p_sm_sgn_3x'])
    # sat
    data['teta_sm_sat_3x'] = pd.DataFrame(data['teta_sm_sat_3x'])
    data['teta_p_sm_sat_3x'] = pd.DataFrame(data['teta_p_sm_sat_3x'])
    # tanh
    data['teta_sm_tanh_3x'] = pd.DataFrame(data['teta_sm_tanh_3x'])
    data['teta_p_sm_tanh_3x'] = pd.DataFrame(data['teta_p_sm_tanh_3x'])
    # general
    data['w_sm_3x'] = pd.DataFrame(data['w_sm_3x'])
    data['t_sm_3x'] = pd.DataFrame(data['t_sm_3x'])
    #----------------------------------------------------------------------------------------------------------------------------------------------------------#
    # to datafram extention data 03#
    # ASM1
    data['teta_sm1_3x'] = pd.DataFrame(data['teta_sm1_3x'])
    data['teta_p_sm1_3x'] = pd.DataFrame(data['teta_p_sm1_3x'])
    data['t_sm1_3x'] = pd.DataFrame(data['t_sm1_3x'])
    # ASM2
    data['teta_sm2_3x'] = pd.DataFrame(data['teta_sm2_3x'])
    data['teta_p_sm2_3x'] = pd.DataFrame(data['teta_p_sm2_3x'])
    data['t_sm2_3x'] = pd.DataFrame(data['t_sm2_3x'])
    
    
    
    return data

def plot_to_base64():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    return base64.b64encode(image_png).decode('utf-8')

def plot_cases(case, data):
    plt.switch_backend('Agg')

    def case01():
        plt.figure(1)
        plt.grid(True)
        plt.plot(data['t_pd_1x'], data['teta_pd_1x'], label='teta_pd', linewidth=1.2, color='blue')
        plt.plot(data['t_do_1x'], data['teta_do_1x'], label='teta_dobc', linewidth=1.2, color='green')
        plt.plot(data['t_es_1x'], data['teta_es_1x'], label='teta_eso', linewidth=1.2, color='red')
        plt.title('teta line plot')
        plt.xlabel('Time label')
        plt.ylabel('teta(t)')
        plt.legend()
        plot1 = plot_to_base64()

        plt.figure(2)
        plt.grid(True)
        plt.plot(data['t_pd_1x'], data['teta_p_pd_1x'], label='teta_p_pd', linewidth=1.2, color='blue')
        plt.plot(data['t_do_1x'], data['teta_p_do_1x'], label='teta_p_dobc', linewidth=1.2, color='green')
        plt.plot(data['t_es_1x'], data['teta_p_es_1x'], label='teta_p_eso', linewidth=1.2, color='red')
        plt.title('teta_dot line plot')
        plt.xlabel('Time label')
        plt.ylabel('teta_p(t)')
        plt.legend()
        plot2 = plot_to_base64()

        return plot1, plot2

    def case02():
        plt.figure(1)
        plt.grid(True)
        plt.plot(data['t_do_1x'], data['teta_do_1x'], label='teta_dobc', linewidth=1.2, color='green')
        plt.plot(data['t_es_1x'], data['teta_es_1x'], label='teta_eso', linewidth=1.2, color='red')
        plt.title('teta line plot')
        plt.xlabel('Time label')
        plt.ylabel('teta(t)')
        plt.legend()
        plot1 = plot_to_base64()

        plt.figure(2)
        plt.grid(True)
        plt.plot(data['t_do_1x'], data['teta_p_do_1x'], label='teta_p_dobc', linewidth=1.2, color='green')
        plt.plot(data['t_es_1x'], data['teta_p_es_1x'], label='teta_p_eso', linewidth=1.2, color='red')
        plt.title('teta_dot line plot')
        plt.xlabel('Time label')
        plt.ylabel('teta_p(t)')
        plt.legend()
        plot2 = plot_to_base64()

        return plot1, plot2
    def case03():
        plt.figure(1)
        plt.grid(True)
        plt.plot(data['t_pd_1x'], data['teta_pd_1x'], label='teta_pd', linewidth=1.2, color='blue')
        plt.plot(data['t_do_1x'], data['teta_do_1x'], label='teta_dobc', linewidth=1.2, color='green')
        plt.title('teta line plot')
        plt.xlabel('Time label')
        plt.ylabel('teta(t)')
        plt.legend()
        plot1 = plot_to_base64()

        plt.figure(2)
        plt.grid(True)
        plt.plot(data['t_pd_1x'], data['teta_p_pd_1x'], label='teta_p_pd', linewidth=1.2, color='blue')
        plt.plot(data['t_do_1x'], data['teta_p_do_1x'], label='teta_p_dobc', linewidth=1.2, color='green')
        plt.title('teta_dot line plot')
        plt.xlabel('Time label')
        plt.ylabel('teta_p(t)')
        plt.legend()
        plot2 = plot_to_base64()

        return plot1, plot2
    
    def case04():
        plt.figure(1)
        plt.grid(True)
        plt.plot(data['t_pd_1x'], data['teta_pd_1x'], label='teta_pd', linewidth=1.2, color='blue')
        plt.plot(data['t_es_1x'], data['teta_es_1x'], label='teta_eso', linewidth=1.2, color='red')
        plt.title('teta line plot')
        plt.xlabel('Time label')
        plt.ylabel('teta(t)')
        plt.legend()
        plot1 = plot_to_base64()

        plt.figure(2)
        plt.grid(True)
        plt.plot(data['t_pd_1x'], data['teta_p_pd_1x'], label='teta_p_pd', linewidth=1.2, color='blue')
        plt.plot(data['t_es_1x'], data['teta_p_es_1x'], label='teta_p_eso', linewidth=1.2, color='red')
        plt.title('teta_dot line plot')
        plt.xlabel('Time label')
        plt.ylabel('teta_p(t)')
        plt.legend()
        plot2 = plot_to_base64()

        return plot1, plot2

    def case05():
        plt.figure(1)
        plt.grid(True)
        plt.plot(data['t_pd_1x'], data['teta_pd_1x'], label='teta_pd', linewidth=1.2, color='blue')
        plt.title('teta line plot')
        plt.xlabel('Time label')
        plt.ylabel('teta(t)')
        plt.legend()
        plot1 = plot_to_base64()

        plt.figure(2)
        plt.grid(True)
        plt.plot(data['t_pd_1x'], data['teta_p_pd_1x'], label='teta_p_pd', linewidth=1.2, color='blue')
        plt.title('teta_dot line plot')
        plt.xlabel('Time label')
        plt.ylabel('teta_p(t)')
        plt.legend()
        plot2 = plot_to_base64()

        return plot1, plot2

    def case06():
        plt.figure(1)
        plt.grid(True)
        plt.plot(data['t_es_1x'], data['teta_es_1x'], label='teta_eso', linewidth=1.2, color='red')
        plt.title('teta line plot')
        plt.xlabel('Time label')
        plt.ylabel('teta(t)')
        plt.legend()
        plot1 = plot_to_base64()

        plt.figure(2)
        plt.grid(True)
        plt.plot(data['t_es_1x'], data['teta_p_es_1x'], label='teta_p_eso', linewidth=1.2, color='red')
        plt.title('teta_dot line plot')
        plt.xlabel('Time label')
        plt.ylabel('teta_p(t)')
        plt.legend()
        plot2 = plot_to_base64()

        return plot1, plot2

    def case07():
        plt.figure(1)
        plt.grid(True)
        plt.plot(data['t_do_1x'], data['teta_do_1x'], label='teta_dobc', linewidth=1.2, color='green')
        plt.title('teta line plot')
        plt.xlabel('Time label')
        plt.ylabel('teta_P(t)')
        plt.legend()
        plot1 = plot_to_base64()

        plt.figure(2)
        plt.grid(True)
        plt.plot(data['t_do_1x'], data['teta_p_do_1x'], label='teta_p_dobc', linewidth=1.2, color='green')
        plt.title('teta_dot line plot')
        plt.xlabel('Time label')
        plt.ylabel('teta_p(t)')
        plt.legend()
        plot2 = plot_to_base64()

        return plot1, plot2

    cases = {
        '1': case01,
        '2': case02,
        '3': case03,
        '4': case04,
        '5': case05,
        '6': case06,
        '7': case07,
    }
    return cases[case]()

def plot_disturbances(data):
    plt.switch_backend('Agg')

    plt.figure(3)
    plt.grid(True)
    plt.plot(data['t_do_1x'], data['w_do_1x'], label='external_disturbance', linewidth=1.2, color='cyan')
    plt.title('External_disturbance plot')
    plt.xlabel('Time label')
    plt.ylabel('w(t)')
    plt.legend()
    external_disturbance = plot_to_base64()

    plt.figure(4)
    plt.grid(True)
    plt.plot(data['t_do_1x'], data['w_do_1x'], label='external_disturbance', linewidth=1.2, color='black')
    plt.title('Internal_disturbance plot')
    plt.xlabel('Time label')
    plt.ylabel('d0(t)')
    plt.legend()
    internal_disturbance = plot_to_base64()

    return external_disturbance, internal_disturbance

# End of Section One Logic #

# Start of Section Two Logic #
def disturbances2(data):
    plt.figure(5) #External disturbance 
    colors = ['red', 'blue', 'green']
    labels = ['w_x', 'w_y', 'w_z']
    #plt.figure(figsize=(10, 6))
    for i in range(3):
        plt.plot(data['t_pd_3x'], data['w_pd_3x'][i], color=colors[i], label=labels[i])
    plt.xlabel('Time')
    plt.ylabel('external disturbance')
    plt.title('Plot of external disturbances with the respect of time')
    plt.legend()
    plt.grid(True)
    external_disturbance = plot_to_base64()

    plt.figure(6) #Internal disturbance 
    colors = ['red', 'blue', 'green']
    labels = ['d_x', 'd_y', 'd_z']
    #plt.figure(figsize=(10, 6))
    for i in range(3):
        plt.plot(data['t_pd_3x'], data['d_pd_3x'][i], color=colors[i], label=labels[i])
    plt.xlabel('Time')
    plt.ylabel('Internal disturbance')
    plt.title('Plot of Internal disturbances with the respect of time')
    plt.legend()
    plt.xlim(0, 60)
    plt.grid(True)
    internal_disturbance = plot_to_base64()

    return external_disturbance, internal_disturbance
    

def SM(sm,data):
    # sm_plot1 = None
    # sm_plot2 = None
    if sm==1: # sign
        plt.figure(1) 
        colors = ['red', 'blue', 'green']
        labels = ['Yaw', 'Pitch', 'Roll']
        for i in range(3):
            plt.plot(data['t_sm_3x'], data['teta_sm_sgn_3x'][i], color=colors[i], label=labels[i])
        plt.xlabel('Time')
        plt.ylabel('Angles')
        plt.title('Plot of Angles_sm_sign with the respect of time')
        plt.legend()
        plt.xlim(0, 250)
        plt.grid(True)
        sm_plot1 = plot_to_base64()

        plt.figure(2) 
        colors = ['red', 'blue', 'green']
        labels = ['Yaw', 'Pitch', 'Roll']
        for i in range(3):
            plt.plot(data['t_sm_3x'],data['teta_p_sm_sgn_3x'][i], color=colors[i], label=labels[i])
        plt.xlabel('Time')
        plt.ylabel('velocity')
        plt.title('Plot of velocity_sm_sign with the respect of time')
        plt.legend()
        plt.xlim(0, 250)
        plt.grid(True)
        sm_plot2 = plot_to_base64()

        return sm_plot1, sm_plot2

    elif sm==2: # sat
        plt.figure(1) 
        colors = ['red', 'blue', 'green']
        labels = ['Yaw', 'Pitch', 'Roll']
        #plt.figure(figsize=(10, 6))
        for i in range(3):
            plt.plot(data['t_sm_3x'], data['teta_sm_sat_3x'][i], color=colors[i], label=labels[i])
        plt.xlabel('Time')
        plt.ylabel('Angles')
        plt.title('Plot of Angles_sm_sat with the respect of time')
        plt.legend()
        plt.xlim(0, 250)
        plt.grid(True)
        sm_plot1 = plot_to_base64()


        plt.figure(2) 
        colors = ['red', 'blue', 'green']
        labels = ['Yaw', 'Pitch', 'Roll']
        #plt.figure(figsize=(10, 6))
        for i in range(3):
            plt.plot(data['t_sm_3x'], data['teta_p_sm_sat_3x'][i], color=colors[i], label=labels[i])
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.title('Plot of Velocity_sm_sat with the respect of time')
        plt.legend()
        plt.xlim(0, 250)
        plt.grid(True)
        sm_plot2 = plot_to_base64()

        return sm_plot1, sm_plot2   
    
    elif sm==3: # tanh
        plt.figure(1) 
        colors = ['red', 'blue', 'green']
        labels = ['Yaw', 'Pitch', 'Roll']
        #plt.figure(figsize=(10, 6))
        for i in range(3):
            plt.plot(data['t_sm_3x'], data['teta_sm_tanh_3x'][i], color=colors[i], label=labels[i])
        plt.xlabel('Time')
        plt.ylabel('Angles')
        plt.title('Plot of Angles_sm_tanh with the respect of time')
        plt.legend()
        plt.xlim(0, 250)
        plt.grid(True)
        sm_plot1 = plot_to_base64()

        plt.figure(2) 
        colors = ['red', 'blue', 'green']
        labels = ['Yaw', 'Pitch', 'Roll']
        #plt.figure(figsize=(10, 6))
        for i in range(3):
            plt.plot(data['t_sm_3x'], data['teta_p_sm_tanh_3x'][i], color=colors[i], label=labels[i])
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.title('Plot of Velocity_sm_tanh with the respect of time')
        plt.legend()
        plt.xlim(0, 250)
        plt.grid(True)
        sm_plot2 = plot_to_base64()

        return sm_plot1, sm_plot2

def plot_second_cases(second_case, data):
    plt.switch_backend('Agg')

    def second_case01():# SM + PD (M1 & M2)
        
        plt.figure(3) 
        colors = ['red', 'blue', 'green']
        labels = ['Yaw', 'Pitch', 'Roll']
        #plt.figure(figsize=(10, 6))
        for i in range(3):
            plt.plot(data['t_pd_3x'], data['teta_pd_3x'][i], color=colors[i], label=labels[i])
        plt.xlabel('Time')
        plt.ylabel('Angles')
        plt.title('Plot of Angles_pd with the respect of time')
        plt.legend()
        plt.xlim(0, 150)
        plt.grid(True)
        case2_plot1 = plot_to_base64()

        plt.figure(4) 
        colors = ['red', 'blue', 'green']
        labels = ['Yaw', 'Pitch', 'Roll']
        #plt.figure(figsize=(10, 6))
        for i in range(3):
            plt.plot(data['t_pd_3x'], data['teta_p_pd_3x'][i], color=colors[i], label=labels[i])
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.title('Plot of Velocity_pd with the respect of time')
        plt.legend()
        plt.xlim(0, 60)
        plt.grid(True)
        case2_plot2 = plot_to_base64()

        return case2_plot1, case2_plot2 
    
    def second_case02(): #SM + ASM1 (M1 & M3)
        # show the RMS value of each method (fhadok les careau)

        plt.figure(3) #ASM1
        colors = ['red', 'blue', 'green']
        labels = ['Yaw', 'Pitch', 'Roll']
        #plt.figure(figsize=(10, 6))
        for i in range(3):
            plt.plot(data['t_sm1_3x'], data['teta_sm1_3x'][i], color=colors[i], label=labels[i])
        plt.xlabel('Time')
        plt.ylabel('Angles')
        plt.title('Plot of Angles_asm1 with the respect of time')
        plt.legend()
        plt.xlim(0, 250)
        plt.grid(True)
        case2_plot1 = plot_to_base64()

        plt.figure(4) #ASM1
        colors = ['red', 'blue', 'green']
        labels = ['Yaw', 'Pitch', 'Roll']
        #plt.figure(figsize=(10, 6))
        for i in range(3):
            plt.plot(data['t_sm1_3x'], data['teta_p_sm1_3x'][i], color=colors[i], label=labels[i])
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.title('Plot of Velocity_asm1 with the respect of time')
        plt.legend()
        plt.xlim(0, 150)
        plt.grid(True)
        case2_plot2 = plot_to_base64()

        return case2_plot1, case2_plot2 
    
    def second_case03(): #SM +ASM2 (M1 & M4)
        # show the RMS value of each method (fhadok les careau)
        plt.figure(3) #ASM2
        colors = ['red', 'blue', 'green']
        labels = ['Yaw', 'Pitch', 'Roll']
        #plt.figure(figsize=(10, 6))
        for i in range(3):
            plt.plot(data['t_sm2_3x'], data['teta_sm2_3x'][i], color=colors[i], label=labels[i])
        plt.xlabel('Time')
        plt.ylabel('Angles')
        plt.title('Plot of Angles_asm2 with the respect of time')
        plt.legend()
        plt.xlim(0, 250)
        plt.grid(True)
        case2_plot1 = plot_to_base64()

        plt.figure(4) #ASM2
        colors = ['red', 'blue', 'green']
        labels = ['Yaw', 'Pitch', 'Roll']
        #plt.figure(figsize=(10, 6))
        for i in range(3):
            plt.plot(data['t_sm2_3x'], data['teta_p_sm2_3x'][i], color=colors[i], label=labels[i])
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.title('Plot of Velocity_asm2 with the respect of time')
        plt.legend()
        plt.xlim(0, 150)
        plt.grid(True)
        case2_plot2 = plot_to_base64()

        return case2_plot1, case2_plot2 
    
    def second_case04():# PD + ASM1 (M2 & M3)
    # show the RMS value of each method (fhadok les careau)
        plt.figure(1) # PD
        colors = ['red', 'blue', 'green']
        labels = ['Yaw', 'Pitch', 'Roll']
        #plt.figure(figsize=(10, 6))
        for i in range(3):
            plt.plot(data['t_pd_3x'], data['teta_pd_3x'][i], color=colors[i], label=labels[i])
        plt.xlabel('Time')
        plt.ylabel('Angles')
        plt.title('Plot of Angles_pd with the respect of time')
        plt.legend()
        plt.xlim(0, 150)
        plt.grid(True)
        sm_plot1 = plot_to_base64()

        plt.figure(2) 
        colors = ['red', 'blue', 'green']
        labels = ['Yaw', 'Pitch', 'Roll']
        #plt.figure(figsize=(10, 6))
        for i in range(3):
            plt.plot(data['t_pd_3x'], data['teta_p_pd_3x'][i], color=colors[i], label=labels[i])
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.title('Plot of Velocity_pd with the respect of time')
        plt.legend()
        plt.xlim(0, 60)
        plt.grid(True)
        sm_plot2 = plot_to_base64()

        plt.figure(3) #ASM1
        colors = ['red', 'blue', 'green']
        labels = ['Yaw', 'Pitch', 'Roll']
        #plt.figure(figsize=(10, 6))
        for i in range(3):
            plt.plot(data['t_sm1_3x'], data['teta_sm1_3x'][i], color=colors[i], label=labels[i])
        plt.xlabel('Time')
        plt.ylabel('Angles')
        plt.title('Plot of Angles_asm1 with the respect of time')
        plt.legend()
        plt.xlim(0, 250)
        plt.grid(True)
        case2_plot1 = plot_to_base64()

        plt.figure(4) #ASM1
        colors = ['red', 'blue', 'green']
        labels = ['Yaw', 'Pitch', 'Roll']
        #plt.figure(figsize=(10, 6))
        for i in range(3):
            plt.plot(data['t_sm1_3x'], data['teta_p_sm1_3x'][i], color=colors[i], label=labels[i])
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.title('Plot of Velocity_asm1 with the respect of time')
        plt.legend()
        plt.xlim(0, 150)
        plt.grid(True)
        case2_plot2 = plot_to_base64()

        return sm_plot1 , sm_plot2 ,case2_plot1, case2_plot2
    # ------------------------------------------------------------------------------------- #
    def second_case05():# PD + ASM2 (M2 & M4)
        # show the RMS value of each method (fhadok les careau)
        plt.figure(1) # PD
        colors = ['red', 'blue', 'green']
        labels = ['Yaw', 'Pitch', 'Roll']
        #plt.figure(figsize=(10, 6))
        for i in range(3):
            plt.plot(data['t_pd_3x'], data['teta_pd_3x'][i], color=colors[i], label=labels[i])
        plt.xlabel('Time')
        plt.ylabel('Angles')
        plt.title('Plot of Angles_pd with the respect of time')
        plt.legend()
        plt.xlim(0, 250)
        plt.grid(True)
        sm_plot1 = plot_to_base64()

        plt.figure(2) 
        colors = ['red', 'blue', 'green']
        labels = ['Yaw', 'Pitch', 'Roll']
        #plt.figure(figsize=(10, 6))
        for i in range(3):
            plt.plot(data['t_pd_3x'], data['teta_p_pd_3x'][i], color=colors[i], label=labels[i])
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.title('Plot of Velocity_pd with the respect of time')
        plt.legend()
        plt.xlim(0, 150)
        plt.grid(True)
        sm_plot2 = plot_to_base64()

        plt.figure(3) #ASM2
        colors = ['red', 'blue', 'green']
        labels = ['Yaw', 'Pitch', 'Roll']
        #plt.figure(figsize=(10, 6))
        for i in range(3):
            plt.plot(data['t_sm2_3x'], data['teta_sm2_3x'][i], color=colors[i], label=labels[i])
        plt.xlabel('Time')
        plt.ylabel('Angles')
        plt.title('Plot of Angles_asm2 with the respect of time')
        plt.legend()
        plt.xlim(0, 250)
        plt.grid(True)
        case2_plot1 = plot_to_base64()

        plt.figure(4) #ASM2
        colors = ['red', 'blue', 'green']
        labels = ['Yaw', 'Pitch', 'Roll']
        #plt.figure(figsize=(10, 6))
        for i in range(3):
            plt.plot(data['t_sm2_3x'], data['teta_p_sm2_3x'][i], color=colors[i], label=labels[i])
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.title('Plot of Velocity_asm2 with the respect of time')
        plt.legend()
        plt.xlim(0, 150)
        plt.grid(True)
        case2_plot2 = plot_to_base64()

        return sm_plot1 , sm_plot2 ,case2_plot1, case2_plot2
    # ---------------------------------------------------------------------------------------------------------------- #
    def second_case06():# ASM1 + ASM2 (M3 & M4)
        # show the RMS value of each method (fhadok les careau)
        plt.figure(1) #ASM1
        colors = ['red', 'blue', 'green']
        labels = ['Yaw', 'Pitch', 'Roll']
        #plt.figure(figsize=(10, 6))
        for i in range(3):
            plt.plot(data['t_sm1_3x'], data['teta_sm1_3x'][i], color=colors[i], label=labels[i])
        plt.xlabel('Time')
        plt.ylabel('Angles')
        plt.title('Plot of Angles_asm1 with the respect of time')
        plt.legend()
        plt.xlim(0, 250)
        plt.grid(True)
        sm_plot1 = plot_to_base64()

        plt.figure(2) #ASM1
        colors = ['red', 'blue', 'green']
        labels = ['Yaw', 'Pitch', 'Roll']
        #plt.figure(figsize=(10, 6))
        for i in range(3):
            plt.plot(data['t_sm1_3x'], data['teta_p_sm1_3x'][i], color=colors[i], label=labels[i])
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.title('Plot of Velocity_asm1 with the respect of time')
        plt.legend()
        plt.xlim(0, 150)
        plt.grid(True)
        sm_plot2 = plot_to_base64()

        plt.figure(3) #ASM2
        colors = ['red', 'blue', 'green']
        labels = ['Yaw', 'Pitch', 'Roll']
        #plt.figure(figsize=(10, 6))
        for i in range(3):
            plt.plot(data['t_sm2_3x'], data['teta_sm2_3x'][i], color=colors[i], label=labels[i])
        plt.xlabel('Time')
        plt.ylabel('Angles')
        plt.title('Plot of Angles_asm2 with the respect of time')
        plt.legend()
        plt.xlim(0, 250)
        plt.grid(True)
        case2_plot1 = plot_to_base64()

        plt.figure(4) #ASM2
        colors = ['red', 'blue', 'green']
        labels = ['Yaw', 'Pitch', 'Roll']
        #plt.figure(figsize=(10, 6))
        for i in range(3):
            plt.plot(data['t_sm2_3x'], data['teta_p_sm2_3x'][i], color=colors[i], label=labels[i])
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.title('Plot of Velocity_asm2 with the respect of time')
        plt.legend()
        plt.xlim(0, 150)
        plt.grid(True)
        case2_plot2 = plot_to_base64()

        return sm_plot1 , sm_plot2 ,case2_plot1, case2_plot2 
    # ---------------------------------------------------------------------------------------------------------------- #



    second_cases = {
        '8': second_case01 ,
        '9': second_case02 ,
        '10': second_case03 ,
        '11': second_case04 ,
        '12': second_case05 ,
        '13': second_case06 ,
    }
    return second_cases[second_case]()

# Simulation Part (Rendering)
def simulation(request):
    if request.method == 'POST':
        #section 1 post data and conditions
        pd_control = 'pd_control' in request.POST
        dobc_control = 'dobc_control' in request.POST
        eso_control = 'eso_control' in request.POST
        extra = 'extra' in request.POST

        data = load_data()
       
        
        if pd_control and dobc_control and eso_control:
            case = '1'
        elif dobc_control and eso_control:
            case = '2'
        elif pd_control and dobc_control:
            case = '3'
        elif pd_control and eso_control:
            case = '4'
        elif pd_control:
            case = '5'
        elif eso_control:
            case = '6'
        elif dobc_control:
            case = '7'
        else:
            case = None
        
        #section 2 post data and conditions 

        tanh = 'tanh' in request.POST
        sign = 'sign' in request.POST
        sat = 'sat' in request.POST
        sm = None
        if sign :
            rms1=0.03523
            sm=1
        if sat :
            rms1=0.03166
            sm=2
        if tanh :
            rms1=0.01202
            sm=3

        pd_dobc_control = 'pd_dobc_control' in request.POST
        asm1 = 'asm1' in request.POST
        asm2 = 'asm2' in request.POST
        extra2 = 'extra2' in request.POST

        if pd_dobc_control :
            second_case = '8'
            rms2=0.05875
        elif asm1 and not asm2 :
            second_case = '9'
            rms2=0.1231
        elif asm2 and not asm1 :
            second_case = '10'
            rms2=0.00671
        elif pd_dobc_control and asm1 :
            second_case = '11'
            rms1=0.05875
            rms2=0.1231


        elif pd_dobc_control and asm2 :
            second_case = '12'
            rms1=0.05875
            rms2=0.00671
        elif asm1 and asm2 :
            second_case = '13'
            rms1=0.1231
            rms2=0.00671

        else:
            second_case= None
            rms1=None
            rms2=None

        #section 3 post data and conditions

        system1=None
        system2=None
        system3=None
        free=None
        closed=None

        fm = 'fm' in request.POST
        cl = 'cl' in request.POST

        sys1 = 'sys1' in request.POST
        sys2 = 'sys2' in request.POST
        sys3 = 'sys3' in request.POST

        if  sys1 and fm and not cl :
            system1 = True
            free = True
            closed = False
        elif sys1 and not fm and cl :
            system1 = True
            free = False
            closed = True
        elif sys1 and fm and cl :
            system1 = True 
            free = True
            closed = True

        elif  sys2 and fm and not cl :
            system2 = True
            free = True
            closed = False
        elif sys2 and not fm and cl :
            system2 = True
            free = False
            closed = True
        elif sys2 and fm and cl :
            system2 = True 
            free = True
            closed = True
        
        elif  sys3 and fm and not cl :
            system3 = True
            free = True
            closed = False
        elif sys3 and not fm and cl :
            system3 = True
            free = False
            closed = True
        elif sys3 and fm and cl :
            system3 = True 
            free = True
            closed = True
    



            
            

        context = {}
        if case:
            context['case_plot1'], context['case_plot2'] = plot_cases(case, data)

        if extra:
            context['disturbances'] = plot_disturbances(data)

        if extra2:
            context['disturbances2'] = disturbances2(data)

        if rms1 and rms2:
            context['rms1']=rms1
            context['rms2']=rms2

        if second_case and sm :
            
            context['sm_plot1'], context['sm_plot2'] = SM(sm , data)
            context['case2_plot1'], context['case2_plot2'] = plot_second_cases(second_case, data)
        elif second_case and not sm:
                print(second_case)
                context['sm_plot1'], context['sm_plot2'] ,context['case2_plot1'], context['case2_plot2'] = plot_second_cases(second_case, data)

        if (system1):
            context['system1'] = system1
            context['free'] = free
            context['closed'] = closed
        elif(system2):
            context['system2'] = system2
            context['free'] = free
            context['closed'] = closed
        elif(system3):
            context['system3'] = system3
            context['free'] = free
            context['closed'] = closed
        else:
            pass
        

        return render(request, 'simulation.html', context)

    return render(request, 'simulation.html')

