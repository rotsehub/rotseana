#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 21:16:04 2025

@author: jacobjuvan
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import subprocess
import tkinter as tk
import argparse
from matplotlib.widgets import Button, TextBox, Slider, RadioButtons

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox, QLineEdit, QRadioButton, QComboBox, QSpinBox, QFormLayout, QSlider, QDialog, QHBoxLayout, QLabel, QCheckBox
from PyQt5.QtCore import Qt

# establish an instance for matplotlib to reference when opening popup windows
app = QApplication.instance()
if not app:  # If there's no QApplication instance, create one
    app = QApplication(sys.argv)
matplotlib.rcParams["font.size"] = 6



# PHASER function fits data to the phase plot
def phaser( lightcurve, period, epoch ):
    phased = list()
    #this first loop adjusts the mjd of each observation to fit to the plot by subtracting the earliest date
    for obs in lightcurve:
        phase = (obs[0] - epoch) / period
        new = [phase % 1, obs[1], obs[2]]
        phased.append(new)
    output = list()
    #this second loop duplicates the adjusted data, shifted horizontally +1 phase
    for i in phased:
        output.append(i)
        tp = [i[0] + 1, i[1], i[2]]
        output.append(tp)
    return output



# FIND_X2 function finds the x2 value of the data, fit to either a sine curve or sech-sec curve
def find_X2(x_val, y_obs, y_exp, phased_data, model_selected):
    #find the x2 value of the data, with expected data fitted to 1 cycle/phase sine curve
    if (model_selected == 'Sine, 1 Cycle per Phase') or (model_selected == 'Sine, 2 Cycles per Phase'):
        x_valadj = list()
        y_obsadj = list()
        y_expadj = list()
        error = list()
        for index in range(len(x_val)):
            if 0 <= x_val[index] and x_val[index] < 1:
                x_valadj.append(x_val[index])
                y_obsadj.append(y_obs[index])
                y_expadj.append(y_exp[index])
                error.append(phased_data[index][2])
        df = len(x_valadj) - 1
        x2_list = list()
        for i in range(len(x_valadj)):
            x2 = ( (y_expadj[i] - y_obsadj[i])**2 ) / (error[i]**2)
            x2_list.append(x2)
        output_x2 = sum(x2_list) / df
        return output_x2
    #find the x2 value of the datas extrema, with expected data fitted to 2 cycles/phase sine curve or sech-sec curve
    if (model_selected == 'Hyperbolic-Secant-Secant') or (model_selected == 'Absolute Sine'):
        x_min = 0.25 + arg_current['Model X-Shift'] + 0.0001
        while x_min > 1:
            x_min = x_min - 1
        while x_min < 0:
            x_min = x_min + 1
        x_min1_low = x_min - arg_current['Chi Width']/2
        x_min1_up = x_min + arg_current['Chi Width']/2
        x_min2_low = x_min1_low + 0.5
        x_min2_up = x_min1_up + 0.5
        if x_min1_low < 0:
            x_min1_low = x_min1_low + 1
            x_min1_up = x_min1_up + 1
        elif x_min1_low > 1:
            x_min1_low = x_min1_low - 1
            x_min1_up = x_min1_up - 1
        if x_min2_low < 0:
            x_min2_low = x_min2_low + 1
            x_min2_up = x_min2_up + 1
        elif x_min2_low > 1:
            x_min2_low = x_min2_low - 1
            x_min2_up = x_min2_up - 1

        min1_l.set_xdata([x_min1_low])
        min1_r.set_xdata([x_min1_up])
        min2_l.set_xdata([x_min2_low])
        min2_r.set_xdata([x_min2_up])
        
        x_l = list()
        y_exp_l = list()
        y_obs_l = list()
        error_l = list()
        x_r = list()
        y_exp_r = list()
        y_obs_r = list()
        error_r = list()
        x_m = list()
        y_exp_m = list()
        y_obs_m = list()
        error_m = list()
        for x_pos in x_val:
            #gather data of left minima
            if ((x_min1_low <= x_pos) and (x_pos <= x_min1_up)):
                index = x_val.index(x_pos)
                y_obs_l1 = y_obs[index]
                y_exp_l1 = y_exp[index]
                error_l1 = phased_data[index][2]
                x_l.append(x_pos)
                y_obs_l.append(y_obs_l1)
                y_exp_l.append(y_exp_l1)
                error_l.append(error_l1)
            #gather data of right minima
            elif ((x_min2_low <= x_pos) and (x_pos <= x_min2_up)):
                index = x_val.index(x_pos)
                y_obs_r1 = y_obs[index]
                y_exp_r1 = y_exp[index]
                error_r1 = phased_data[index][2]
                x_r.append(x_pos)
                y_obs_r.append(y_obs_r1)
                y_exp_r.append(y_exp_r1)
                error_r.append(error_r1)
            #gather data of maximas
            else:
                index = x_val.index(x_pos)
                y_obs_m1 = y_obs[index]
                y_exp_m1 = y_exp[index]
                error_m1 = phased_data[index][2]
                x_m.append(x_pos)
                y_obs_m.append(y_obs_m1)
                y_exp_m.append(y_exp_m1)
                error_m.append(error_m1)
        if model_selected == 'Hyperbolic-Secant-Secant':
            parameters = 4
        elif model_selected == 'Absolute Sine':
            parameters = 3
        df_l = len(x_l) - parameters
        df_r = len(x_r) - parameters
        df_m = len(x_m) - parameters
        left_x2_list = list()
        right_x2_list = list()
        middle_x2_list = list()
        for i in range(len(y_exp_l)):
            x2 = ( (y_exp_l[i] - y_obs_l[i])**2 ) / ((error_l[i])**2)
            left_x2_list.append(x2)
        for j in range(len(y_exp_r)):
            x2 = ( (y_exp_r[j] - y_obs_r[j])**2 ) / ((error_r[j])**2)
            right_x2_list.append(x2)
        for k in range(len(y_exp_m)):
            x2 = ( (y_exp_m[k] - y_obs_m[k])**2 ) / ((error_m[k])**2)
            middle_x2_list.append(x2)
        left_x2 = sum(left_x2_list) / df_l
        right_x2 = sum(right_x2_list) / df_r
        middle_x2 = sum(middle_x2_list) / df_m
        return [left_x2, right_x2, middle_x2]    
    


# EVALUATOR determines whether or not the input value can be evaluated, allows textboxes to be cleared and reset
def evaluator(text):
    try:
        num = eval(text)
        if isinstance(num, int) or isinstance(num, float):
            return True
        else:
            return False
    except:
        return False
    


# COLLECT_FILE collects user input, SAVE_DATA saves lcvis output depending on user input
def collect_file():
    dlg = QDialog()
    dlg.setWindowTitle('Save Menu')
    
    def ask_overwrite():
        cwd = os.getcwd()
        os.chdir(cwd)
        if filetype_dat.isChecked():
            extension = '_lcvis.dat'
        elif filetype_pkl.isChecked():
            if plots_lcvis.isChecked() and plots_chisqbin.isChecked() and bin_text.get_text() != '':
                extension = '_lcvis_ALLfits.pkl'
            elif plots_lcvis.isChecked() and plots_chisqbin.isChecked() and bin_text.get_text() == '':
                extension = '_lcvis_LCVfits.pkl'
            elif plots_lcvis.isChecked() and not plots_chisqbin.isChecked():
                extension = '_lcvis_LCVfits.pkl'
            elif (plots_chisqbin.isChecked() and bin_text.get_text() != '') and not plots_lcvis.isChecked():
                extension = '_lcvis_CSBTfits.pkl'
            else:
                extension = '_lcvis_NOfits.pkl'
        savename_extension.setText(extension)

        filename = savename_box.text()
        if filename == '':
            filename = 'unnamed'
        file_path = './' + f'{filename}' + f'{extension}'
        if os.path.isfile(file_path):
            savename_overwrite.setVisible(True)
        else:
            savename_overwrite.setVisible(False)
            savename_overwrite.setChecked(False)
    
    savename_box = QLineEdit(dlg)
    savename_box.setFixedSize(200,25)
    savename_box.textEdited.connect(ask_overwrite)
    savename_extension = QLabel()
    savename_overwrite = QCheckBox("Overwrite?", dlg)
    savename_overwrite.setVisible(False)
    savename_layout = QHBoxLayout()
    savename_layout.addWidget(savename_box)
    savename_layout.addWidget(savename_extension)
    savename_layout.addWidget(savename_overwrite)
    
    def show_optionals():
        ask_overwrite()
        plots_label.setVisible(True)
        plots_lcvis.setVisible(True)
        plots_chisqbin.setVisible(True)
    
    def hide_optionals():
        ask_overwrite()
        plots_label.setVisible(False)
        plots_lcvis.setVisible(False)
        plots_chisqbin.setVisible(False)
        
    
    filetype_dat = QRadioButton(".dat", dlg)
    filetype_dat.clicked.connect(hide_optionals)
    filetype_pkl = QRadioButton(".pkl", dlg)
    filetype_pkl.clicked.connect(show_optionals)
    filetype_dat.setChecked(True)
    filetype_layout = QHBoxLayout()
    filetype_layout.addWidget(filetype_dat)
    filetype_layout.addWidget(filetype_pkl)
    
    plots_label = QLabel("Optional Fits")
    plots_lcvis = QCheckBox("Lcvis Fits", dlg)
    plots_lcvis.clicked.connect(ask_overwrite)
    plots_chisqbin = QCheckBox("CSBT Fits", dlg)
    plots_chisqbin.clicked.connect(ask_overwrite)
    plots_layout = QHBoxLayout()
    plots_layout.addWidget(plots_lcvis)
    plots_layout.addWidget(plots_chisqbin)
    hide_optionals()
    
    options = [None, None, None, None]
    clicked = [None]
    
    def cancel_clicked():
        clicked[0] = 'Cancel'
        dlg.close()
        
    def save_clicked():
        clicked[0] = 'Save'
        save_lcvis_models = False
        save_chisqbin_models = False
        if filetype_dat.isChecked():
            extension = '_lcvis.dat'
        elif filetype_pkl.isChecked():
            if plots_lcvis.isChecked() and plots_chisqbin.isChecked() and bin_text.get_text() != '':
                extension = '_lcvis_ALLfits.pkl'
                save_lcvis_models = True
                save_chisqbin_models = True
            elif plots_lcvis.isChecked() and not (plots_chisqbin.isChecked() or bin_text.get_text() != ''):
                extension = '_lcvis_LCVfits.pkl'
                save_lcvis_models = True
                save_chisqbin_models = False
            elif (plots_chisqbin.isChecked() and bin_text.get_text() != '') and not plots_lcvis.isChecked():
                extension = '_lcvis_CSBTfits.pkl'
                save_lcvis_models = False
                save_chisqbin_models = True
            else:
                extension = '_lcvis_NOfits.pkl'
                save_lcvis_models = False
                save_chisqbin_models = False
        
        filename = savename_box.text()
        if filename == '':
            filename = 'unnamed'
        file_path = './' + f'{filename}' + f'{extension}'
        if not savename_overwrite.isChecked():
            i = 0
            old_filename = filename
            while os.path.isfile(file_path):
                filename = f'{old_filename}_dupe{i}'
                file_path = './' + f'{filename}' + f'{extension}'
                i += 1
        
        options[0] = filename
        options[1] = extension
        options[2] = save_lcvis_models
        options[3] = save_chisqbin_models
        
        dlg.close()
    
    save_button = QPushButton('Save',dlg)
    save_button.setFixedSize(100,25)
    save_button.clicked.connect(save_clicked)
    
    cancel_button = QPushButton('Cancel',dlg)
    cancel_button.setFixedSize(100,25)
    cancel_button.clicked.connect(dlg.close)
    
    form = QFormLayout()
    form.addRow("Filename", savename_layout)
    form.addRow("File Type", filetype_layout)
    form.addRow(plots_label, plots_layout)
    form.addRow(save_button, cancel_button)
    
    dlg.setLayout(form)
    dlg.exec_()
    
    return options
    
    
    
def save_data(event):
    options = collect_file()
    cwd = os.getcwd()
    os.chdir(cwd)
    if None not in options:
        matplotlib.rcParams["font.size"] = 7
        filename = f'{options[0]}' + f'{options[1]}'
        extension = options[1]
        save_lcvis_models = options[2]
        save_chisqbin_models = options[3]
        if extension == '_lcvis.dat':
            phased_init = phaser(data,arg_current['Period'],data[0][0])
            phased = list()
            for obs in phased_init:
                if obs[0] < 1:
                    phased.append(obs)
            phased.sort()
            np.savetxt(filename, phased, fmt = '%.11f')
            save_complete = QMessageBox()
            save_complete.setIcon(QMessageBox.Information)
            save_complete.setText(f'Saved {filename} to {cwd}.')
            save_complete.setStandardButtons(QMessageBox.Ok)
            save_complete.exec_()
            
        else:
            ymax, ymin = ax.get_ylim()
            fig2, ax2 = plt.subplots()
            plt.get_current_fig_manager().set_window_title('Saved Lcvis Output for ' + name)
            fig2.suptitle('Lcvis Plot for ' + name)
            ax2.set_ylim(top = ymax, bottom = ymin)
            plt.gca().invert_yaxis()
            plt.xlabel('Phase')
            plt.ylabel('Magnitude')
            plt.subplots_adjust(left=0.06,right=0.7,bottom=0.1,top=0.95)
            
            save = phaser(data,arg_current['Period'],data[0][0])
            
            x1 = list()
            y1 = list()
            yerror = list()
            for i in save:
                x1.append(i[0])
                y1.append(i[1])
                yerror.append(i[2])
            q2 = plt.errorbar(x1,y1,yerr = yerror,fmt=".",c='k', markersize=7, elinewidth=1)
            
            legend = [q2]
            legend_titles = ['Observed Data with Error']
            fig2.text(0.73,0.75, per_text.get_text())
            fig2.text(0.85,0.75, mag_text.get_text())
            
            if save_lcvis_models:
                x_curve = curve.get_xdata()
                y_curve = curve.get_ydata()
                curve2, = plt.plot(x_curve, y_curve, c='orange', lw=1)
                
                x_scatter = x1
                y_scatter = list()
                for x in x_scatter:
                    y = equations(x)[current_selection['Model']]
                    y_scatter.append(y)
                data_expected2 = plt.scatter(x_scatter, y_scatter, c='silver', s=40, marker = '.')
                fig2.text(0.73,0.87, eq_text.get_text())
                fig2.text(0.73,0.6, chi_text.get_text())
                
                legend.extend([data_expected2, curve2])
                legend_titles.extend(['Expected Data on Lcvis Model', 'Lcvis Curve Model'])
                
                if current_selection['Model'] in ['Absolute Sine', 'Hyperbolic-Secant-Secant']:
                    min1_l2 = plt.axvline(x=min1_l.get_xdata(), color = 'red', ls = '-', lw = 1)
                    min1_r2 = plt.axvline(x=min1_r.get_xdata(), color = 'red', ls = '-', lw = 1)
                    min2_l2 = plt.axvline(x=min2_l.get_xdata(), color = 'blue', ls = '-', lw = 1)
                    min2_r2 = plt.axvline(x=min2_r.get_xdata(), color = 'blue', ls = '-', lw = 1)
                
                    legend.extend([min1_l2, min2_l2])
                    legend_titles.extend(['Lcvis Min 1', 'Lcvis Min 2'])
                
            if save_chisqbin_models and bin_text.get_text() != '':
                min1_fit1x, min1_fit1y = min1_fit1.get_xdata(), min1_fit1.get_ydata()
                min2_fit2x, min2_fit2y = min2_fit2.get_xdata(), min2_fit2.get_ydata()
                min1_fit2x, min1_fit2y = min1_fit2.get_xdata(), min1_fit2.get_ydata()
                min2_fit1x, min2_fit1y = min2_fit1.get_xdata(), min2_fit1.get_ydata()
                
                min1_data_scat = min1_data.get_offsets()
                min1_datax, min1_datay = min1_data_scat[:,0], min1_data_scat[:,1]
                min2_data_scat = min2_data.get_offsets()
                min2_datax, min2_datay = min2_data_scat[:,0], min2_data_scat[:,1]
                
                min1_fit1s, = plt.plot(min1_fit1x, min1_fit1y, color = 'darkkhaki', linewidth = 2, zorder = 5)
                min2_fit1s, = plt.plot(min2_fit1x, min2_fit1y, color = 'darkkhaki', linewidth = 1.5, linestyle = '--', zorder = 6)
                min2_fit2s, = plt.plot(min2_fit2x, min2_fit2y, color = 'green', linewidth = 2, zorder = 5)
                min1_fit2s, = plt.plot(min1_fit2x, min1_fit2y, color = 'green', linewidth = 1.5, linestyle = '--', zorder = 6)
                min1_datas = plt.scatter(min1_datax, min1_datay, c = 'khaki', s=4, marker = '.', zorder = 4)
                min2_datas = plt.scatter(min2_datax, min2_datay, c = 'lightgreen', s=4, marker = '.', zorder = 4)
                
                max_fitx, max_fity = max_fit.get_xdata(), max_fit.get_ydata()
                max_data_scat = max_data.get_offsets()
                max_datax, max_datay = max_data_scat[:,0], max_data_scat[:,1]
                
                max_fits, = plt.plot(max_fitx, max_fity, color = 'saddlebrown', linewidth = 2, zorder = 5)
                max_datas = plt.scatter(max_datax, max_datay, c = 'chocolate', s=2, marker = '.', zorder = 4)
                
                
                legend.extend([min1_datas, min2_datas, max_datas, min1_fit1s, min2_fit1s, min2_fit2s, min1_fit2s, max_fits])
                legend_titles.extend(['CSBT P Data', 'CSBT S Data', 'CSBT Max Data', 'CSBT P Fit', 'CSBT P Fit on S Data', 'CSBT S Fit', 'CSBT S Fit on P Data', 'CSBT Max Fit'])
            
                fig2.text(0.73,0.1, bin_text.get_text())
            
            plt.legend(legend, legend_titles, loc = 'lower right')
            
            pickle.dump(fig2, open(f'{filename}', 'wb'))
            fig2.show()
            matplotlib.rcParams["font.size"] = 6
            plt.figure(fig.number)
            
            save_complete = QMessageBox()
            save_complete.setIcon(QMessageBox.Information)
            save_complete.setText(f'Saved {filename} to {cwd}.')
            save_complete.setStandardButtons(QMessageBox.Ok)
            save_complete.exec_()
    


# RUN_CHISQBINTEST runs the csbt.py program with input parameters   
def run_chisqbintest(event):
    dlg = QDialog()
    dlg.setWindowTitle('Input csbt.py parameters')
    
    numpoints_box = QSpinBox(dlg)
    numpoints_box.setMinimum(2)
    numpoints_box.setMaximum(50)
    numpoints_box.setValue(20)
    
    def update_label(value):
        window_label.setText('{:.2f}'.format(round(value * 0.01, 2)))
    
    window_layout = QHBoxLayout()
    window_label = QLabel("0.10")
    window_slider = QSlider(Qt.Horizontal)
    window_slider.setMinimum(int(0.05 / 0.01))
    window_slider.setMaximum(int(0.30 / 0.01))
    window_slider.setValue(int(0.10 / 0.01))
    window_slider.valueChanged.connect(update_label)
    
    window_layout.addWidget(window_label)
    window_layout.addWidget(window_slider)
    
    method_box = QComboBox(dlg)
    method_box.addItems(['move_avg', 'low_mags'])
    
    clicked = [None]
    
    def cancel_clicked():
        clicked[0] = 'Cancel'
        dlg.close()
        
    def run_clicked():
        clicked[0] = 'Run'
        dlg.close()
    
    run_button = QPushButton('Run',dlg)
    run_button.clicked.connect(run_clicked)
    
    cancel_button = QPushButton('Cancel',dlg)
    cancel_button.clicked.connect(cancel_clicked)
    
    form = QFormLayout()
    form.addRow("Number of Points (2-50)", numpoints_box)
    form.addRow("Window", window_layout)
    form.addRow("Method", method_box)
    form.addRow(run_button, cancel_button)
    
    dlg.setLayout(form)
    dlg.exec_()
    
    if clicked[0] == 'Run':
        numpoints = numpoints_box.text()
        window = window_label.text()
        method = method_box.currentText()
            
        phased_init = phaser(data,arg_current['Period'],data[0][0])
        chibin_data = list()
        for obs in phased_init:
            if obs[0] < 1:
                chibin_data.append(obs)
        chibin_data.sort()
        
        try: 
            results = subprocess.run(['python3.10', os.path.dirname(__file__) + "/csbt.py", f"{str(chibin_data)}", "-n", f"{numpoints}", "-w", f"{window}", "-m", f"{method}", "-c", "lcs", "-p", "false"], stdout=subprocess.PIPE)
            parameters = pickle.loads(results.stdout)
        
            d2 = np.vstack((parameters["Pri_Min_Phases"],parameters["Pri_Min_Mags"]))
            min1_data.set_offsets(d2.T)
            
            d2 = np.vstack((parameters["Sec_Min_Phases"],parameters["Sec_Min_Mags"]))
            min2_data.set_offsets(d2.T)
            
            parab_width = float(window)
            x_min1 = np.linspace(parameters["Pri_Phase"] - parab_width, parameters["Pri_Phase"] + parab_width, 100)
            x_min2 = np.linspace(parameters["Sec_Phase"] - parab_width, parameters["Sec_Phase"] + parab_width, 100)
            x_max = np.linspace(0,2,1000)
            
            min1_fit1.set_xdata(x_min1)
            min1_fit1y = []
            for x in x_min1:
                min1_fit1y.append(parameters["Pri_A"] * (x - parameters["Pri_Phase"])**2 + parameters["Pri_Mag"])
            min1_fit1.set_ydata(min1_fit1y)
            
            min2_fit2.set_xdata(x_min2)
            min2_fit2y = []
            for x in x_min2:
                min2_fit2y.append(parameters["Sec_A"] * (x - parameters["Sec_Phase"])**2 + parameters["Sec_Mag"])
            min2_fit2.set_ydata(min2_fit2y)
            min1_fit2.set_xdata(x_min1)
            min1_fit2.set_ydata(min2_fit2y)
            min2_fit1.set_xdata(x_min2)
            min2_fit1.set_ydata(min1_fit1y)
            
            y_max = []
            for x in x_max:
                y_max.append(parameters["Max_FitAmp"] * (np.sin(2*np.pi*(x-parameters["Max_FitPhase"])))**2 + parameters["Max_FitMag"])
            max_fit.set_xdata(x_max)
            max_fit.set_ydata(y_max)
            
            d2 = np.vstack((parameters["Max_Phase"], parameters["Max_Mag"]))
            max_data.set_offsets(d2.T)
            
            bin_text.set_text(display_binary_params(parameters, numpoints, window, method))
            
            plt.draw()
            
            run_complete = QMessageBox()
            run_complete.setIcon(QMessageBox.Information)
            run_complete.setText('csbt.py ran succesfully.')
            run_complete.setStandardButtons(QMessageBox.Ok)
            run_complete.exec_()
        
        except:
            show_error('CSBT could not run properly. Try a different period, pethod, or window size.')

         
        

# CREATE_PLOT is used to plot the initial data as well as create a saved pkl plot when desired
def create_plot(data, period):
    magmin = data[0][1]
    magmin_error = data[0][2]
    magmin_lim = magmin - magmin_error - 0.05
    magmax = data[0][1]
    magmax_error = data[0][2]
    magmax_lim = magmax + magmax_error + 0.05
    for obs in data:
        if obs[1] < magmin:
            magmin = obs[1]
            magmin_error = obs[2]
            magmin_lim = magmin - magmin_error - 0.05
        if obs[1] > magmax:
            magmax = obs[1]
            magmax_error = obs[2]
            magmax_lim = magmax + magmax_error + 0.05
    
    phased_data = phaser(data,period,data[0][0])

    # plot initial data
    phases = list()
    mags = list()
    mag_errors = list()
    for i in phased_data:
        phases.append(i[0])
        mags.append(i[1])
        mag_errors.append(i[2])

    # plot initial sin curve, adjusted to data
    x_curve = np.arange(0,2,0.0001)

    y_datashift = 0.5 * (max(mags)+min(mags))

    y1_avg = np.mean(mags)
    y1_sorted = mags.copy()
    y1_sorted.sort(reverse=False)
    y1_mins = y1_sorted[0:10]
    y1_sorted.sort(reverse=True)
    y1_maxs = y1_sorted[0:10]
    y1_maxavg = sum(y1_maxs) / len(y1_maxs)
    y1_minavg = sum(y1_mins) / len(y1_mins)
    amp_data = round(((y1_avg - y1_minavg) + (y1_maxavg - y1_avg)) / 2,6)
    sin_width = 2 * np.pi

    y_curve = np.sin(sin_width * x_curve) * amp_data + y_datashift
    
    # plot initial sine scatter, adjusted to data. used for x2 calculation
    x_scatter = phases
    y_scatter = list()
    for x_val in x_scatter:
        y_val = np.sin(sin_width * x_val) * (amp_data) + y_datashift
        y_scatter.append(y_val)
    
    return magmax_lim, magmin_lim, phased_data, phases, mags, mag_errors, x_curve, y_curve, y_scatter, amp_data, y_datashift



# REDRAW_PLOT adjusts the lightcurve, curve, expected data accordingly
def redraw_plot():
    lightcurve = phaser(data, arg_current['Period'], data[0][0])
    phases = list()
    mags = list()
    errors = list()
    for obs in lightcurve:
        phases.append(obs[0])
        mags.append(obs[1])
        errors.append(obs[2])
    q[0].set_xdata(phases)
    for bar in q[2]:
        segments = bar.get_segments()
        
        new_segments = []
        for i in range(len(segments)):
            segments[i][0][0] = phases[i]
            segments[i][1][0] = phases[i]
            
            new_segments.append([[segments[i][0][0], segments[i][0][1]], [segments[i][1][0], segments[i][1][1]]])
        bar.set_segments(new_segments)
    
    curve.set_ydata(equations(x_curve)[current_selection['Model']])
    y_scatter = list()
    for x in phases:
        y = equations(x)[current_selection['Model']]
        y_scatter.append(y)
    d2 = np.vstack((phases,y_scatter))
    data_expected.set_offsets(d2.T)
    
    per_text.set_text(display_period(round(arg_current['Period'],6)))
    param_text.set_text(display_parameters(current_selection['Model'], current_selection['Argument']))
    eq_text.set_text(display_equation(current_selection['Model']))
    chi_text.set_text(display_chi(current_selection['Model'], find_X2(phases, mags, y_scatter, lightcurve, current_selection['Model'])))
    mag_text.set_text(display_mag_limits(ax))
    plt.draw()
    
    

# MAG_RANGER function allows adjustment of the magnitude axis
def mag_ranger(text):
    if evaluator(text):
        mag_range = eval(text)
        if mag_range > 0:
            mag_setrange = eval(text) / 2
            mag_max, mag_min = ax.get_ylim()
            mag_midpt = (mag_max + mag_min) / 2
            mag_min = mag_midpt - mag_setrange
            mag_max = mag_midpt + mag_setrange
            ax.set_ylim(top = mag_min, bottom = mag_max)
        else:
            ymax, ymin = ax.get_ylim()
            old = str(round(ymax-ymin,6))
            show_error('Magnitude range cannot be less than or equal to 0'
                       + f' (entered {round(mag_range,6)}). Old value of {round(old,6)} preserved.')
            magrange_box.set_val('')
            return
        redraw_plot()
    elif text != '':
        tk.messagebox.showerror(title='Error', message='Only integers and numbers can be entered.')
        magrange_box.set_val('')
        


# ADDER function to manually add to period, phase shift, amplitude, x-shift, and y-shift, with given stepsize
def adder(arg_selected):
    new_val = arg_current[current_selection['Argument']] + arg_stepsize[current_selection['Argument']]
    if new_val <= arg_max[current_selection['Argument']]:
        arg_current[current_selection['Argument']] = new_val
        slider.set_val(new_val)
        redraw_plot()



# SUBTRACTOR function to manually subtract from period, amplitude, model x-shift, and model y-shift, with given stepsize
def subtractor(arg_selected):
    new_val = arg_current[current_selection['Argument']] - arg_stepsize[current_selection['Argument']]
    if new_val >= arg_min[current_selection['Argument']]:
        arg_current[current_selection['Argument']] = new_val
        slider.set_val(new_val)
        redraw_plot()



# STEPPER function to set stepsize size of period, amplitude, model x-shift, and model y-shift of their respective sliders
def stepper(text):
    if evaluator(text):
        new_stepsize = eval(text)
        if new_stepsize > 0:
            arg_stepsize[current_selection['Argument']] = new_stepsize
            slider.valstep = new_stepsize
            redraw_plot()
        else:
            show_error('Stepsize cannot be less than or equal to 0'
                       + f" (entered {round(new_stepsize,6)}). Old value of {round(arg_stepsize[current_selection['Argument']],6)} preserved.")
            stepbox.set_val('')
    elif text != '':
        show_error('Only integers and numbers can be entered.')
        stepbox.set_val('')



# SETTER function to manually set period, amplitude, model x-shift, and model y-shift
def setter(text):
    if evaluator(text):
        new_set_val = eval(text)
        if current_selection['Argument'] == 'Period' or current_selection['Argument'] == 'Min Extrema Width':
            if new_set_val > 0:
                arg_current[current_selection['Argument']] = new_set_val
                slider.set_val(new_set_val)
                redraw_plot()
            else:
                show_error(f"{current_selection['Argument']} cannot be less than or equal to 0"
                           + f" (entered {round(new_set_val,6)}). Old value of {round(arg_current[current_selection['Argument']],6)} preserved.")
                setbox.set_val('')
        else:
            arg_current[current_selection['Argument']] = new_set_val
            slider.set_val(new_set_val)
            redraw_plot()
    elif text != '':
        show_error('Only integers and numbers can be entered.')
        setbox.set_val('')



# MAXXER function sets the maximum value of the slider for the current argument
def maxxer(text):
    if evaluator(text):
        new_arg_max = eval(text)
        if new_arg_max > slider.valmin:
            if current_selection['Argument'] == 'Period' or current_selection['Argument'] == 'Min Extrema Width':
                if new_arg_max > 0:
                    arg_max[current_selection['Argument']] = new_arg_max
                    slider.valmax = new_arg_max
                    slider.ax.set_xlim(slider.valmin,new_arg_max)
                else:
                    show_error(f"{current_selection['Argument']} cannot be less than or equal to 0"
                              + f" (entered {round(new_arg_max,6)}). Old value of {round(arg_max[current_selection['Argument']],6)} preserved.")
                    maxbox.set_val('')
            else:
                arg_max[current_selection['Argument']] = new_arg_max
                slider.valmax = new_arg_max
                slider.ax.set_xlim(slider.valmin,new_arg_max)
        else:
            show_error('Maximum slider value is less than or equal to minimum slider value'
                       + f' (entered {new_arg_max}, max = {slider.valmin}). Old value of {slider.valmax} preserved.')
            maxbox.set_val('')
        redraw_plot()
    elif text != '':
        show_error('Only integers and numbers can be entered.')
        maxbox.set_val('')



# MAG_MAXXER function sets the maximum magnitude displayed on the y-axis
def mag_maxxer(text):
    if evaluator(text):
        ymax, ymin = ax.get_ylim()
        new_ymax = eval(text)
        if new_ymax <= ymin:
            show_error('Maximum magnitude is less than or equal to minimum magnitude'
                       + f' (entered {round(new_ymax,6)}, min = {round(ymin,6)}). Old value of {round(ymax,6)} preserved.')
            magmax_box.set_val('')
            return
        else:
            ax.set_ylim(bottom = new_ymax)
        redraw_plot()
    elif text != '':
        show_error('Only integers and numbers can be entered.')
        magmax_box.set_val('')



# MINNER function sets the minimum value of the slider for the current argument
def minner(text):
    if evaluator(text):
        new_arg_min = eval(text)
        if new_arg_min < slider.valmax:
            if current_selection['Argument'] == 'Period' or current_selection['Argument'] == 'Min Extrema Width':
                if new_arg_min > 0:
                    arg_min[current_selection['Argument']] = new_arg_min
                    slider.valmin = new_arg_min
                    slider.ax.set_xlim(new_arg_min,slider.valmax)
                else:
                    show_error(f"{current_selection['Argument']} cannot be less than or equal to 0"
                              + f" (entered {round(new_arg_min,6)}). Old value of {round(arg_min[current_selection['Argument']],6)} preserved.")
                    minbox.set_val('')
            else:
                arg_min[current_selection['Argument']] = new_arg_min
                slider.valmin = new_arg_min
                slider.ax.set_xlim(new_arg_min,slider.valmax)
        else:
            show_error('Minimum slider value is greater than or equal to maximum slider value'
                       + f' (entered {new_arg_min}, max = {slider.valmax}). Old value of {slider.valmin} preserved.')
            minbox.set_val('')
        redraw_plot()
    elif text != '':
        show_error('Only integers and numbers can be entered.')
        minbox.set_val('')

def mag_minner(text):
    if evaluator(text):
        ymax, ymin = ax.get_ylim()
        new_ymin = eval(text)
        if new_ymin >= ymax:
            show_error('Minimum magnitude is greater than or equal to maximum magnitude'
                       + f' (entered {round(new_ymin,6)}, max = {round(ymax,6)}). Old value of {round(ymin,6)} preserved.')
            magmin_box.set_val('')
        else:
            ax.set_ylim(top = new_ymin)
    elif text != '':
        show_error('Only integers and numbers can be entered.')
        magmin_box.set_val('')



# UPDATE functions to update period, amplitude, model x-shift, and model y-shift from their respective sliders
def updater(val):
    arg_current[current_selection['Argument']] = slider.val
    redraw_plot()



# CHI_UPDATER function to adjust the width of the chi-range
def chi_updater(val):
    arg_current['Chi Width'] = chislider.val
    redraw_plot()



# SELECT functions change the arguments of buttons, sliders, and text boxes, as well as model displayed
def select_argument(arg):
    if arg == 'Min Extrema Width' and current_selection['Model'] != 'Hyperbolic-Secant-Secant':
        show_error('Min Extrema Width can only be adjusted for a '
                   + 'Hyperbolic-Secant-Secant curve. Setting active argument to Period.')
        arg = 'Period'
        argument_radio.set_active(0)
    current_selection['Argument'] = arg
    Badd.label.set_text(f'+{arg}')
    Bsub.label.set_text(f'-{arg}')
    stepbox.label.set_text(f'{arg} Stepsize ')
    setbox.label.set_text(f'Set {arg} ')
    maxbox.label.set_text(f'{arg} Max ')
    minbox.label.set_text(f'{arg} Min ')
    slider.label.set_text(f'{arg}')
    slider.valmax = arg_max[arg]
    slider.valmin = arg_min[arg]
    slider.ax.set_xlim(arg_min[arg],arg_max[arg])
    slider.valstep = arg_stepsize[arg]
    slider.set_val(arg_current[arg])
    clear_boxes()
    redraw_plot()

def select_model(model):
    current_selection['Model'] = model
    curve.set_ydata(equations(x_curve)[model])
    y_scatter = list()
    x_scatter = data_expected.get_offsets()[:,0]
    for x in x_scatter:
        y = equations(x)[model]
        y_scatter.append(y)
    d2 = np.vstack((x_scatter,y_scatter))
    data_expected.set_offsets(d2.T)
    if model == 'Sine, 1 Cycle per Phase' or model == 'Sine, 2 Cycles per Phase':
        min1_l.set_lw(0)
        min1_r.set_lw(0)
        min2_l.set_lw(0)
        min2_r.set_lw(0)
    else:
        min1_l.set_lw(1)
        min1_r.set_lw(1)
        min2_l.set_lw(1)
        min2_r.set_lw(1)
    eq_text.set_text(display_equation(model))
    clear_boxes()
    redraw_plot()



# CLEAR_BOXES function runs when a new argument is selected or when an error occurs
def clear_boxes():
    setbox.set_val('')
    stepbox.set_val('')
    maxbox.set_val('')
    minbox.set_val('')
    magmax_box.set_val('')
    magmin_box.set_val('')
    magrange_box.set_val('')
    
def clear_chibin(event):
    if bin_text.get_text() != '':
        d2 = np.vstack((0,0))
        min1_data.set_offsets(d2.T)
        
        d2 = np.vstack((0,0))
        min2_data.set_offsets(d2.T)
        
        min1_fit1.set_xdata([0])
        min1_fit1.set_ydata([0])
        
        min2_fit2.set_xdata([0])
        min2_fit2.set_ydata([0])
        
        min1_fit2.set_xdata([0])
        min1_fit2.set_ydata([0])
        
        min2_fit1.set_xdata([0])
        min2_fit1.set_ydata([0])
        
        d2 = np.vstack((0,0))
        max_data.set_offsets(d2.T)
        
        max_fit.set_xdata([0])
        max_fit.set_ydata([0])
        
        bin_text.set_text('')
        
        plt.draw()
        
        clear_complete = QMessageBox()
        clear_complete.setIcon(QMessageBox.Information)
        clear_complete.setText('Sucessfully cleared CSBT output.')
        clear_complete.setStandardButtons(QMessageBox.Ok)
        clear_complete.exec_()



# SHOW_ERROR function shows error popup when error encountered
def show_error(message):
    error_box = QMessageBox()
    error_box.setIcon(QMessageBox.Critical)
    error_box.setText(message)
    error_box.setStandardButtons(QMessageBox.Ok)
    
    error_box.exec_()



# DISPLAY functions establish the text for the output
def display_parameters(model_selected, arg_selected):
    parameters_info = ("Curve type = " + model_selected + "          "
                    + arg_selected + " Stepsize = " + str(round(arg_stepsize[arg_selected],6)) + "          "
                    + arg_selected + " Min = " + str(round(arg_min[arg_selected],6)) + "          "
                    + arg_selected + " Max = " + str(round(arg_max[arg_selected],6)) + "          "
                    + "Data centered at magnitude " + str(round(centered,6)))
    return parameters_info

def display_equation(model_selected):
    equations_info = lambda : {'Sine, 1 Cycle per Phase' : (str(round(arg_current['Amplitude'],6) ) + ' * sin[2π * (x - '
                        + str( round(arg_current['Model X-Shift'],6) ) + ')] + ' + str( round(arg_current['Model Y-Shift'] + y_datashift,6))),
                          'Sine, 2 Cycles per Phase' : (str(round(arg_current['Amplitude'],6) ) + ' * sin[4π * (x - '
                                              + str( round(arg_current['Model X-Shift'],6) ) + ')] + ' + str( round(arg_current['Model Y-Shift'] + y_datashift,6))),
                          'Absolute Sine' : (str(round(arg_current['Amplitude'],6) ) + ' * |sin[2π * (x - '
                                              + str( round((arg_current['Model X-Shift'] - 0.25),6) ) + ')]| + ' + str( round(arg_current['Model Y-Shift'] + y_datashift,6))),
                          'Hyperbolic-Secant-Secant' : (str(round(arg_current['Amplitude'],6) ) + ' * sech{' + str( round(arg_current['Min Extrema Width'], 6)) 
                                              + ' * sec[2π * (x - ' + str(round(arg_current['Model X-Shift'],6)) + ')]}\n'
                                              + '                           + ' + str(round(arg_current['Model Y-Shift'] + y_datashift,6)))}
    title = "LCVIS MODEL EQUATION"
    underlined_title = ''.join([char + '\u0332' for char in title])
    
    if model_selected == 'Hyperbolic-Secant-Secant':
        equation_info = (underlined_title + '\n'
                         + 'Model Equation: ' + equations_info()[model_selected] + '\n' 
                         + 'Model Amplitude: ' + str( round(arg_current['Amplitude'],6) ) + '\n'
                         + 'Manually Adjusted Model X-Shift: ' + str( round(arg_current['Model X-Shift'],6) ) + '\n'
                         + 'Manually Adjusted Model Y-Shift: ' + str( round(arg_current['Model Y-Shift'],6) ) + '\n'
                         + 'Manually Adjusted Min Extrema Width: ' + str( round(arg_current['Min Extrema Width'],6)))
    else:
        equation_info = (underlined_title + '\n'
                         + 'Model Equation: ' + equations_info()[model_selected] + '\n' 
                         + 'Model Amplitude: ' + str( round(arg_current['Amplitude'],6) ) + '\n'
                         + 'Manually Adjusted Model X-Shift: ' + str( round(arg_current['Model X-Shift'],6) ) + '\n'
                         + 'Manually Adjusted Model Y-Shift: ' + str( round(arg_current['Model Y-Shift'],6) ))
    return equation_info

def display_period(period):
    title = "PERIOD"
    underlined_title = ''.join([char + '\u0332' for char in title])
    period_info = (underlined_title + '\n'
                   + 'Guess Period: ' + str(per_initial) + '\n'
                   + 'Plot Period: ' + str(round(period,6)))
    return period_info

def display_chi(model, x2):
    title = "LCVIS MODEL CHI-SQUARE"
    underlined_title = ''.join([char + '\u0332' for char in title])
    if model == 'Sine, 1 Cycle per Phase' or model == 'Sine, 2 Cycles per Phase':
        chi_info = (underlined_title + '\n' + 'χ²: ' + str( round(x2,6) ))
    else:
        chi_info = (underlined_title + '\n'
                    + 'Red Minima χ²: ' + str( round(x2[0],6) ) + '\n'
                    + 'Blue Minima χ²: ' + str( round(x2[1],6) ) + '\n'
                    + 'Maxima χ²: ' + str( round(x2[2],6) ) + '\n'
                    + 'Minima χ² Width: ' + str( round(arg_current['Chi Width'],6) ))
    return chi_info

def display_mag_limits(ax):
    ymax, ymin = ax.get_ylim()
    title = "MAGNITUDE LIMITS"
    underlined_title = ''.join([char + '\u0332' for char in title])
    mag_info = (underlined_title + '\n'
                + 'Magnitude Min: ' + str(round(ymin,6)) + '\n'
                + 'Magnitude Max: ' + str(round(ymax,6)) + '\n'
                + 'Magnitude Range: ' + str(round(ymax - ymin,6)))
    return mag_info

def display_binary_params(params, numpoints, window, method):
    title1 = "CHISQBINTEST OUTPUT"
    title2 = "BINARY PARAMETERS"
    underlined_title1 = ''.join([char + '\u0332' for char in title1])
    underlined_title2 = ''.join([char + '\u0332' for char in title2])

    binary_info = (underlined_title1 + '\n'
                   + f"Method: {method}, Number of Points: {numpoints}" + '\n'
                   + f"Window: {window} phase, Period: {round(arg_current['Period'],6)}" + '\n\n'
                   + 'Primary Fit Equation: ' + str(round(params["Pri_A"],6)) + " * (x - " + str(round(params["Pri_Phase"],6)) + ")² + " + str(round(params["Pri_Mag"],6)) + '\n'
                   + 'P Fit, P Data χ²: ' + str(round(params["FitP,MinP"],6)) + '\n'
                   + 'P Fit, P Data Left Half χ²: ' + str(round(params["FitP,MinPL"],6)) + '\n'
                   + 'P Fit, P Data Right Half χ²: ' + str(round(params["FitP,MinPR"],6)) + '\n'
                   + 'P Fit, S Data χ²: ' + str(round(params["FitP,MinS"],6)) + '\n\n'
                   + 'Secondary Fit Equation: ' + str(round(params["Sec_A"],6)) + " * (x - " + str(round(params["Sec_Phase"],6)) + ")² + " + str(round(params["Sec_Mag"],6)) + '\n'
                   + 'S Fit, S Data χ²: ' + str(round(params["FitS,MinS"],6)) + '\n'
                   + 'S Fit, S Data Left Half χ²: ' + str(round(params["FitS,MinSL"],6)) + '\n'
                   + 'S Fit, S Data Right Half χ²: ' + str(round(params["FitS,MinSR"],6)) + '\n'
                   + 'S Fit, P Data χ²: ' + str(round(params["FitS,MinP"],6)) + '\n\n'
                   + 'Max Fit: ' + str(round(params["Max_FitAmp"],6)) + ' * sin[2π * (x - ' + str(round(params["Max_FitPhase"],6)) + ')]² + ' + str(round(params["Max_FitMag"],6)) + '\n'
                   + 'Max Fit χ²: ' + str(round(params["Max_X2"],6)) + '\n\n\n'
                   + underlined_title2 + '\n'
                   + 'Oblateness * sin²(Inclination): ' + str(round(params['ep_sin2i'],6)) + '\n'
                   + 'Effective Temperature Ratio: ' + str(round(params['Teff'],6)) + '\n'
                   + 'Eccentricity: ' + str(round(params['Eccentricity'],6)) + '\n'
                   + 'Periastron: ' + str(round(params['Periastron'],6)))
    '''
    # this is drastically simplified text
    binary_info = (underlined_title1 + '\n'
                   + f"Method: {method}, Number of Points: {numpoints}" + '\n'
                   + f"Window: {window} phase, Period: {round(arg_current['Period'],6)}" + '\n\n'
                   + 'P Fit, P Data χ²: ' + str(round(params["FitP,MinP"],6)) + '\n'
                   + 'S Fit, P Data χ²: ' + str(round(params["FitS,MinP"],6)) + '\n\n'
                   + 'S Fit, S Data χ²: ' + str(round(params["FitS,MinS"],6)) + '\n'
                   + 'P Fit, S Data χ²: ' + str(round(params["FitP,MinS"],6)) + '\n\n'
                   + 'Max Fit χ²: ' + str(round(params["Max_X2"],6)) + '\n\n\n'
                   + underlined_title2 + '\n'
                   + 'Oblateness * sin²(Inclination): ' + str(round(params['ep_sin2i'],6)) + '\n'
                   + 'Effective Temperature Ratio: ' + str(round(params['Teff'],6)) + '\n'
                   + 'Eccentricity: ' + str(round(params['Eccentricity'],6)) + '\n'
                   + 'Periastron: ' + str(round(params['Periastron'],6)))
    '''
    
    return binary_info



# create arguments
parser = argparse.ArgumentParser()
parser.add_argument("indata", help = "Object's unconex data file")
parser.add_argument("--initial_period", "-p0", default = 1, help = "Initial period for plot; use single_phase period on unconex data")
args = parser.parse_args()
indata = args.indata
object_name = os.path.basename(indata)


# gather arguments
data = (np.loadtxt(indata)).tolist()
per_initial = float(args.initial_period)
name = object_name


# create plots
fig, ax = plt.subplots()
plt.xlabel('Phase')
plt.ylabel('Magnitude')
plt.subplots_adjust(left=0.06,right=0.7,bottom=0.3,top=0.95)

magmax_lim, magmin_lim, phased_data, phases, mags, mag_errors, x_curve, y_curve, y_scatter, amp_data, y_datashift = create_plot(data, per_initial)
ax.set_ylim(magmax_lim, magmin_lim)
q = plt.errorbar(phases, mags, yerr=mag_errors, c='k', fmt=".", markersize=7, elinewidth = 1, zorder = 3)
centered = np.mean(mags)
curve, = plt.plot(x_curve, y_curve, c='orange', zorder = 2, lw=1)
data_expected = plt.scatter(phases, y_scatter, c='darkgray', marker = '.', s=40, zorder = 1)

# create dictionaries of initial argument values, maxes, mins, and stepsizes
arg_max = {"Period" : 2, "Amplitude" : round(0.5 * (max(mags)-min(mags)) + 0.5,6), 
           "Model Y-Shift" : 1, "Model X-Shift" : 1, "Min Extrema Width" : 2}
arg_min = {"Period" : 0.000001, "Amplitude" : 0, "Model Y-Shift" : -1,
           "Model X-Shift" : -1, "Min Extrema Width" : 0.000001}
arg_stepsize = {"Period" : 0.000001, "Amplitude" : 0.000001, "Model Y-Shift" : 0.000001,
           "Model X-Shift" : 0.000001, "Min Extrema Width" : 0.000001}
arg_current = {"Period" : per_initial, "Amplitude" : amp_data, "Model Y-Shift" : 0,
           "Model X-Shift" : 0, "Min Extrema Width" : 0.1, "Chi Width" : 0.1}
current_selection = {'Argument' : 'Period', 'Model' : 'Sine, 1 Cycle per Phase'}

equations = lambda x : {'Sine, 1 Cycle per Phase' : (arg_current['Amplitude'] * np.sin(2 * np.pi * (x - arg_current['Model X-Shift'])) + arg_current['Model Y-Shift'] + y_datashift),
                      'Sine, 2 Cycles per Phase' : (arg_current['Amplitude'] * np.sin(4 * np.pi * (x - arg_current['Model X-Shift'])) + arg_current['Model Y-Shift'] + y_datashift),
                      'Absolute Sine' : (-arg_current['Amplitude'] * abs(np.sin(2 * np.pi * (x - arg_current['Model X-Shift'] - 0.25))) + arg_current['Model Y-Shift'] + y_datashift),
                      'Hyperbolic-Secant-Secant' : (-arg_current['Amplitude'] / ( np.cosh((arg_current['Min Extrema Width']) / np.cos(2 * np.pi * (x - arg_current['Model X-Shift']))) ) + arg_current['Model Y-Shift'] + y_datashift)}

# creating minima bound lines for binary minima chi-square analysis
x_min = 0.25 + arg_current['Model X-Shift'] + 0.0001
min1_lx = x_min - arg_current['Chi Width']/2
min1_rx = x_min + arg_current['Chi Width']/2
min2_lx = x_min - arg_current['Chi Width']/2 + 0.5
min2_rx = x_min + arg_current['Chi Width']/2 + 0.5
min1_l = plt.axvline(x=min1_lx, color = 'red', ls = '-', lw = 1)
min1_r = plt.axvline(x=min1_rx, color = 'red', ls = '-', lw = 1)
min2_l = plt.axvline(x=min2_lx, color = 'blue', ls = '-', lw = 1)
min2_r = plt.axvline(x=min2_rx, color = 'blue', ls = '-', lw = 1)

# creating initial curves for chisqbintest
min1_fit1, = plt.plot(0, 0, color = 'darkkhaki', linewidth = 2, zorder = 6)
min2_fit1, = plt.plot(0, 0, color = 'darkkhaki', linewidth = 1.5, linestyle = '--', zorder = 7)
min2_fit2, = plt.plot(0, 0, color = 'green', linewidth = 2, zorder = 6)
min1_fit2, = plt.plot(0, 0, color = 'green', linewidth = 1.5, linestyle = '--', zorder = 7)
min1_data = plt.scatter(0, 0, c = 'khaki', s=4, marker = '.', zorder = 4)
min2_data = plt.scatter(0, 0, c = 'lightgreen', s=4, marker = '.', zorder = 4)
max_fit, = plt.plot(0, 0, color = 'saddlebrown', linewidth = 2, zorder = 5)
max_data = plt.scatter(0, 0, c = 'chocolate', s=2, marker = '.', zorder = 4)

plt.legend([q, data_expected, curve, min1_l, min2_l, min1_data, min2_data, max_data, min1_fit1, min2_fit1, min2_fit2, min1_fit2, max_fit], 
           ['Observed Data with Error', 'Expected Data on Lcvis Model', 'Lcvis Curve Model', 'Lcvis Min 1', 'Lcvis Min 2',
            'CSBT Primary Data', 'CSBT Secondary Data', 'CSBT Max Data', 'CSBT P Fit', 'CSBT P Fit on S Data', 'CSBT S Fit', 'CSBT S Fit on P Data', 'CSBT Max Fit'], loc = 'lower right')

min1_l.set_lw(0)
min1_r.set_lw(0)
min2_l.set_lw(0)
min2_r.set_lw(0)

#create gui for textboxes, buttons, sliders
argument_ax = plt.axes([0.06,0.05,0.10,0.12])
model_ax = plt.axes([0.17,0.05,0.14,0.12])

add = plt.axes([0.32,0.12,0.08,0.03])
sub = plt.axes([0.32,0.07,0.08,0.03])
steptext = plt.axes([0.68,0.12,0.06,0.03])
settext = plt.axes([0.68,0.07,0.06,0.03])
maxtext = plt.axes([0.51,0.12,0.06,0.03])
mintext = plt.axes([0.51,0.07,0.06,0.03])
magrange_ax = plt.axes([0.81,0.025,0.06,0.03])
magmax_ax = plt.axes([0.81,0.125,0.06,0.03])
magmin_ax = plt.axes([0.81,0.075,0.06,0.03])
chiwidth_ax = plt.axes([0.61,0.175,0.25,0.03])
slider_ax = plt.axes([0.1,0.215,0.80,0.03])
save_ax = plt.axes([0.9,0.125,0.055,0.03])
runchibin_ax = plt.axes([0.9,0.075,0.055,0.03])
clearchibin_ax = plt.axes([0.9,0.025,0.055,0.03])


# calculate initial x2
x2_calculated = find_X2(phases, mags, y_scatter, phased_data, current_selection['Model'])


# make textboxes, buttons, sliders    
argument_radio = RadioButtons(ax=argument_ax, labels=('Period', 'Amplitude', 'Model X-Shift', 'Model Y-Shift', 'Min Extrema Width'),active=0,activecolor='red')
model_radio = RadioButtons(ax=model_ax, labels=('Sine, 1 Cycle per Phase', 'Sine, 2 Cycles per Phase', 'Absolute Sine', 'Hyperbolic-Secant-Secant'), active=0,activecolor='red')

Badd = Button(ax=add,label=f"+{current_selection['Argument']}")
Bsub = Button(ax=sub,label=f"-{current_selection['Argument']}")
stepbox = TextBox(steptext,f"{current_selection['Argument']} Stepsize ")
setbox = TextBox(settext,f"Set {current_selection['Argument']} ")
maxbox = TextBox(maxtext,f"{current_selection['Argument']} Max ")
minbox = TextBox(mintext,f"{current_selection['Argument']} Min ")
magrange_box = TextBox(magrange_ax, 'Magnitude Range ')
magmax_box = TextBox(magmax_ax, 'Magnitude Max ')
magmin_box = TextBox(magmin_ax, 'Magnitude Min ')
chislider = Slider(chiwidth_ax, 'Absolute Sine and Hyperbolic-Secant-Secant Only: Minima χ² Width',
                   0.05, 0.3, valinit = arg_current['Chi Width'], valstep = 0.01, color='black')
slider = Slider(slider_ax, f"{current_selection['Argument']}", arg_min[current_selection['Argument']],
                arg_max[current_selection['Argument']], valinit = 0,
                valstep = arg_stepsize[current_selection['Argument']], color='black')
slider.set_val(per_initial)
Bsave = Button(ax=save_ax,label='Save')
Brunchibin = Button(ax=runchibin_ax, label='Run csbt.py')
Bclearchibin = Button(ax=clearchibin_ax, label='Clear CSBT Output')


#give functions to textboxes, buttons, sliders
argument_radio.on_clicked(select_argument)
model_radio.on_clicked(select_model)

Badd_id = Badd.on_clicked(adder)
Bsub_id = Bsub.on_clicked(subtractor)
stepbox_id = stepbox.on_submit(stepper)
setbox_id = setbox.on_submit(setter)
maxbox_id = maxbox.on_submit(maxxer)
minbox_id = minbox.on_submit(minner)
magrange_box.on_submit(mag_ranger)
magmax_box.on_submit(mag_maxxer)
magmin_box.on_submit(mag_minner)
chislider.on_changed(chi_updater)
slider_id = slider.on_changed(updater)
Bsave.on_clicked(save_data)
Brunchibin.on_clicked(run_chisqbintest)
Bclearchibin.on_clicked(clear_chibin)


#create initial titles and texts
plt.get_current_fig_manager().set_window_title('Lcvis Output for ' + name)
fig.suptitle('Lcvis Plot for ' + name)

param_text = fig.text(0.06,0.025, display_parameters(current_selection['Model'], current_selection['Argument']))
eq_text = fig.text(0.73,0.87, display_equation(current_selection['Model']))
per_text = fig.text(0.73,0.8, display_period(per_initial))
chi_text = fig.text(0.73,0.7, display_chi(current_selection['Model'], x2_calculated))
mag_text = fig.text(0.83,0.8, display_mag_limits(ax))
bin_text = fig.text(0.73,0.27,'')

fig.text(0.06, 0.18, "Arguments: ")
fig.text(0.17,0.18, "Models: ")

# show plot
plt.show()
