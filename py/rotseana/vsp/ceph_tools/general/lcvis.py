def lcvis( data, obj_name, period_initial ):

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import pickle
    import tkinter
    from matplotlib.widgets import Button, TextBox, Slider, RadioButtons

    matplotlib.use('TkAgg')

    #these variables are updated throughout inside functions
    global arg_selected
    global x_sin
    global y_sin
    global modelx_manualshift
    global y_sin_shift
    global y_sin_datashift
    global modelx_manualshift
    global modely_manualshift
    global y_parab_shift
    global amp_data
    global amp_manual
    global sin_width
    global curve_type
    global centered
    global q
    global per_max
    global per_min
    global amp_max
    global amp_min
    global modelx_max
    global modelx_min
    global modely_max
    global modely_min
    global slider
    global Badd_id
    global Bsub_id
    global stepbox_id
    global setbox_id
    global maxbox_id
    global minbox_id
    global slider_id




    #PHASER function fits data to the phase plot
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




    #FIND_X2 function finds the x2 value of the data, fit to either a sine curve or absolute value sine curve
    def find_X2(x_val, y_obs, y_exp, phased_data):
        #find the x2 value of the data, with expected data fitted to 1 cycle/phase sine curve
        if (curve_type == 'sine'):
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
        #find the x2 value of the datas extrema, with expected data fitted to 2 cycles/phase sine curve or absolute value sine curve
        if (curve_type == 'abs sine'):
            global amp_data
            global amp_manual
            global modelx_manualshift
            global y_parab_shift
            x_vals = np.arange(0,1,0.001)
            x_min = x_vals[0]
            y_min = -abs(np.sin((sin_width * (x_min - modelx_manualshift))) * (amp_data + amp_manual)) + y_sin_shift
            for x in x_vals:
                y_val = -abs(np.sin((sin_width * (x - modelx_manualshift))) * (amp_data + amp_manual)) + y_sin_shift
                if y_val > y_min:
                    y_min = y_val
                    x_min = x
            
            if x_min < 0.5:
                x_min = x_min + 1
            x_min1_low = x_min - 0.1
            x_min1_up = x_min + 0.1
            x_min2_low = x_min1_low - 0.1
            x_min2_up = x_min1_up + 0.1
            
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
            df_l = len(x_l) - 1
            df_r = len(x_r) - 1
            df_m = len(x_m) - 1
            left_x2_list = list()
            right_x2_list = list()
            middle_x2_list = list()
            for i in range(df_l):
                x2 = ( (y_exp_l[i] - y_obs_l[i])**2 ) / ((error_l[i])**2)
                left_x2_list.append(x2)
            for j in range(df_r):
                x2 = ( (y_exp_r[j] - y_obs_r[j])**2 ) / ((error_r[j])**2)
                right_x2_list.append(x2)
            for k in range(df_m):
                x2 = ( (y_exp_m[k] - y_obs_m[k])**2 ) / ((error_m[k])**2)
                middle_x2_list.append(x2)
            left_x2 = sum(left_x2_list) / df_l
            right_x2 = sum(right_x2_list) / df_r
            middle_x2 = sum(middle_x2_list) / df_m
            return [left_x2, right_x2, middle_x2]




    #REDRAW_PLOT function updates the plot
    def redraw_plot():
        global q
        #update data points
        phased = phaser(data,per,data[0][0])
        x1 = list()
        y1 = list()
        yerror = list()
        for j in phased:
            x1.append(j[0])
            y1.append(j[1])
            yerror.append(j[2])
        q.remove()
        q = ax.errorbar(x1,y1,yerr=yerror,fmt="o",c='tab:blue',elinewidth=1)

        x_sin_curve = np.arange(0,2,0.001)

        y_sin_datashift = 0.5 * (max(y1)+min(y1))
        y_sin_shift = modely_manualshift + y_sin_datashift
        
        x_sin_scatter = x1
        y_sin_scatter = list()

        if curve_type == 'sine':
            #update sine curve

            y_sin_curve = np.sin((sin_width * (x_sin_curve - modelx_manualshift))) * (amp_data + amp_manual) + y_sin_shift
            curve.set_ydata(y_sin_curve)

            #update sine scatter
            for x_val in x_sin_scatter:
                y_val = np.sin((sin_width * (x_val - modelx_manualshift))) * (amp_data + amp_manual) + y_sin_shift
                y_sin_scatter.append(y_val)
            d2 = np.vstack((x_sin_scatter,y_sin_scatter))
            data_expected.set_offsets(d2.T)

            #recalculate x2
            x2_calculated = find_X2(x_sin_scatter, y1, y_sin_scatter, phased)

        if curve_type == 'abs sine':
            #update absolute sine curve
            
            y_sin_curve = -abs(np.sin((sin_width * (x_sin_curve - modelx_manualshift))) * (amp_data + amp_manual)) + y_sin_shift
            curve.set_ydata(y_sin_curve)

            #update sine scatter
            for x_val in x_sin_scatter:
                y_val = -abs(np.sin((sin_width * (x_val - modelx_manualshift))) * (amp_data + amp_manual)) + y_sin_shift
                y_sin_scatter.append(y_val)
            d2 = np.vstack((x_sin_scatter,y_sin_scatter))
            data_expected.set_offsets(d2.T)

            #recalculate x2
            x2_calculated = find_X2(x_sin_scatter, y1, y_sin_scatter, phased)

        #update figure titles
        
        if curve_type == 'sine':
            eq_info.set_text('Model Equation: ' + str( round(amp_data + amp_manual,6) ) + ' * sin[' + str( round(sin_width,6) )
                                + ' * (x - ' + str( round(modelx_manualshift,6) ) + ')] + ' + str( round(y_sin_shift,6)) + '\n' 
                                + 'Model Amplitude: ' + str( round(amp_data,6) ) + '\n'
                                + 'Manually Adjusted Model X-Shift: ' + str( round(modelx_manualshift,6) ) + '\n'
                                + 'Manually Adjusted Model Y-Shift: ' + str( round(modely_manualshift,6) ) )
            chi_info.set_text('X2: ' + str( round(x2_calculated,6) ))
        elif curve_type == 'abs sine':
            eq_info.set_text('Model Equation: ' + str( round(amp_data + amp_manual,6) ) + ' * |sin[' + str( round(sin_width,6) )
                                + ' * (x - ' + str( round(modelx_manualshift,6) ) + ')]| + ' + str( round(y_sin_shift,6)) + '\n' 
                                + 'Model Amplitude: ' + str( round(amp_data,6) ) + '\n'
                                + 'Manually Adjusted Model X-Shift: ' + str( round(modelx_manualshift,6) ) + '\n'
                                + 'Manually Adjusted Model Y-Shift: ' + str( round(modely_manualshift,6) ) )
            chi_info.set_text('First Minima X2: ' + str( round(x2_calculated[0],6) ) + '\n'
                                + 'Second Minima X2: ' + str( round(x2_calculated[1],6) ) + '\n'
                                + 'Maxima X2: ' + str( round(x2_calculated[2],6) ))
        
        ymax, ymin = ax.get_ylim()
        centered = (ymax + ymin) / 2
        text.set_text("Curve type = " + curve_type + "          "
                        + arg_selected + " Stepsize = " + str(round(slider.valstep,6)) + "          "
                        + arg_selected + " Min = " + str(round(slider.valmin,6)) + "          "
                        + arg_selected + " Max = " + str(round(slider.valmax,6)) + "          "
                        + "Plot centered at magnitude " + str(round(centered,6)))
        
        per_info.set_text('Guess Period: ' + str(period_initial) + '\n' + 'Plot Period: ' + str(round(per,6)))
        
        ymax, ymin = ax.get_ylim()
        mag_info.set_text('Magnitude Min: ' + str(round(ymin,6)) + '\n'
                            + 'Magnitude Max: ' + str(round(ymax,6)) + '\n'
                            + 'Magnitude Range: ' + str(round(ymax - ymin,6)))
        
        fig.canvas.draw_idle()
        
        
        
        
    #EVALUATOR determines whether or not the input value can be evaluated, allows textboxes to be cleared and reset
    def evaluator(text):
        try:
            eval(text)
            return True
        except:
            return False
        
        
        
        
    #SAVE functions allow plots to be saved as .png and .pkl files
    def save_pkl(event):
        show_model = tkinter.messagebox.askyesnocancel(title='Save?', message="Save plot with model?")
        if show_model == True or show_model == False:
            cwd = os.getcwd()
            os.chdir(cwd)
            filename = f'{object_name}_lcvisNoModel.pkl'
            ymax, ymin = ax.get_ylim()
            fig2, ax2 = plt.subplots()
            plt.gca().invert_yaxis()
            plt.xlabel('Phase')
            plt.ylabel('Magnitude')
            plt.subplots_adjust(left=0.1,bottom=0.4,top=0.95)
            
            save = phaser(data,per,data[0][0])
            
            x1 = list()
            y1 = list()
            yerror = list()
            for i in save:
                x1.append(i[0])
                y1.append(i[1])
                yerror.append(i[2])
            #p = plt.scatter(x1,y1)
            p = plt.errorbar(x1,y1,yerr = yerror,fmt="o",elinewidth=1)
            
            if show_model == True:
                filename = f'{object_name}_lcvis.pkl'
                x_sin_scatter = x1
                y_sin_scatter2 = list()
                y_sin_datashift = 0.5 * (max(y1)+min(y1))
                y_sin_shift = modely_manualshift + y_sin_datashift
                
                if curve_type == 'sine':
                    #update sine curve
        
                    y_sin_curve2 = np.sin((sin_width * (x_sin_curve - modelx_manualshift))) * (amp_data + amp_manual) + y_sin_shift
                    curve2, = plt.plot(x_sin_curve,y_sin_curve2,'g-')
        
                    #update sine scatter
                    for x_val in x_sin_scatter:
                        y_val = np.sin((sin_width * (x_val - modelx_manualshift))) * (amp_data + amp_manual) + y_sin_shift
                        y_sin_scatter2.append(y_val)
                    data_expected2 = plt.scatter(x_sin_scatter,y_sin_scatter2,c='tab:orange')
                    
                    x2_calculated = find_X2(x_sin_scatter, y1, y_sin_scatter, save)
                    
                    fig2.text(0.1,0.27,'Model Equation: ' + str( round(amp_data + amp_manual,6) ) + ' * sin[' + str( round(sin_width,6) )
                                        + ' * (x - ' + str( round(modelx_manualshift,6) ) + ')] + ' + str( round(y_sin_shift,6)) + '\n' 
                                        + 'Model Amplitude: ' + str( round(amp_data,6) ) + '\n'
                                        + 'Manually Adjusted Model X-Shift: ' + str( round(modelx_manualshift,6) ) + '\n'
                                        + 'Manually Adjusted Model Y-Shift: ' + str( round(modely_manualshift,6) ) )
                    fig2.text(0.65,0.27, 'X2: ' + str( round(x2_calculated,6) ))
                    
        
                if curve_type == 'abs sine':
                    #update absolute sine curve
                    
                    y_sin_curve2 = -abs(np.sin((sin_width * (x_sin_curve - modelx_manualshift))) * (amp_data + amp_manual)) + y_sin_shift
                    curve2, = plt.plot(x_sin_curve,y_sin_curve2,'g-')
        
                    #update sine scatter
                    for x_val in x_sin_scatter:
                        y_val = -abs(np.sin((sin_width * (x_val - modelx_manualshift))) * (amp_data + amp_manual)) + y_sin_shift
                        y_sin_scatter2.append(y_val)
                    data_expected2 = plt.scatter(x_sin_scatter,y_sin_scatter2,c='tab:orange')
                    
                    x2_calculated = find_X2(x_sin_scatter, y1, y_sin_scatter, save)
                    
                    fig2.text(0.1,0.27,'Model Equation: ' + str( round(amp_data + amp_manual,6) ) + ' * |sin[' + str( round(sin_width,6) )
                                        + ' * (x - ' + str( round(modelx_manualshift,6) ) + ')]| + ' + str( round(y_sin_shift,6)) + '\n' 
                                        + 'Model Amplitude: ' + str( round(amp_data,6) ) + '\n'
                                        + 'Manually Adjusted Model X-Shift: ' + str( round(modelx_manualshift,6) ) + '\n'
                                        + 'Manually Adjusted Model Y-Shift: ' + str( round(modely_manualshift,6) ) )
                    fig2.text(0.65,0.27, 'First Minima X2: ' + str( round(x2_calculated[0],6) ) + '\n'
                                        + 'Second Minima X2: ' + str( round(x2_calculated[1],6) ) + '\n'
                                        + 'Maxima X2: ' + str( round(x2_calculated[2],6) ))
                
                plt.legend([p, data_expected2, curve2], ['Observed Data with Error', 'Expected Data on Model', 'Curve Model'],
                           loc = 'lower right')
                
            elif show_model == False:
                plt.legend([p], ['Observed Data with Error'], loc = 'lower right')
            
            fig2.text(0.53,0.27, 'Guess Period: ' + str(period_initial) + '\n' + 'Plot Period: ' + str(round(per,6)))
            
            fig2.suptitle('Lcvis Plot for ' + obj_name)
            
            ax2.set_ylim(top = ymin, bottom = ymax)

            fig2.text(0.4,0.27, 'Magnitude Min: ' + str(round(ymin,6)) + '\n'
                                + 'Magnitude Max: ' + str(round(ymax,6)) + '\n'
                                + 'Magnitude Range: ' + str(round(ymax - ymin,6)))
            
            pickle.dump(fig2, open(f'{filename}', 'wb'))
            preview_plot = tkinter.messagebox.askyesno(title='Show plot?', message="Would you like to view the saved plot?")
            if preview_plot:
                fig2.show()
                
            tkinter.messagebox.showinfo(title='Figure Saved', message=f"A pickled plot of the light curve named {filename}"
                                        + f" has been saved in {cwd}"
                                        + "\n\n" + "Plot cannot be changed when opening with unpickle.py")
            plt.figure(fig.number)
            
        elif show_model == None:
            tkinter.messagebox.showinfo(title='Save Cancelled', message='Cancelled saving plot.')

        


    #FREQUENCY functions to change the frequency of the displayed sine curve and scatter, or to change to absolute value sine
    def frequency_1x():
        global sin_width
        global curve_type
        global amp_manual
        curve_type = 'sine'
        sin_width = 2 * np.pi
        redraw_plot()

    def frequency_2x():
        global sin_width
        global curve_type
        global amp_manual
        curve_type = 'sine'
        sin_width = 4 * np.pi
        redraw_plot()

    def abssine_curve():
        global sin_width
        global curve_type
        global amp_manual
        curve_type = 'abs sine'
        sin_width = 2 * np.pi
        redraw_plot()

            

    
    #ARG_SELECTOR function allows argument selector to select argument
    def arg_selector(label):
        global arg_selected
        arg_selected = label
        if label == 'Period':
            select_period()
        elif label == 'Amplitude':
            select_amplitude()
        elif label == 'Model Y-Shift':
            select_modelyshift()
        elif label == 'Model X-Shift':
            select_modelxshift()
        setbox.set_val('')
        stepbox.set_val('')
        maxbox.set_val('')
        minbox.set_val('')
        magmax_box.set_val('')
        magmin_box.set_val('')
        magrange_box.set_val('')
            
    def model_selector(label):
        if label == 'Sine, 1 Cycle per Phase':
            frequency_1x()
        elif label == 'Sine, 2 Cycles per Phase':
            frequency_2x()
        elif label == 'Absolute Value Sine':
            abssine_curve()
        setbox.set_val('')
        stepbox.set_val('')
        maxbox.set_val('')
        minbox.set_val('')
        magmax_box.set_val('')
        magmin_box.set_val('')
        magrange_box.set_val('')
            
            
            
            
    #MAG_RANGER function allows adjustment of the magnitude axis
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
                tkinter.messagebox.showwarning(title='Error', message='Magnitude range cannot be less than or equal to 0'
                                               + f' (entered {mag_range}). Old value of {old} preserved.')
                magrange_box.set_val('')
                return
            redraw_plot()



            
    #ADDER functions to manually add to period, phase shift, amplitude, x-shift, and y-shift, with given stepsize
    def adder(event):
        global arg_selected
        if arg_selected == 'Period':
            global per
            global stepsize_per
            temp_per = per + stepsize_per
            if temp_per > 0:
                per = temp_per
                slider.set_val(per)
        elif arg_selected == 'Amplitude':
            global amp_manual
            global stepsize_amp
            amp_manual = amp_manual + stepsize_amp
            slider.set_val(amp_data + amp_manual)
        elif arg_selected == 'Model X-Shift':
            global modelx_manualshift
            global stepsize_modelxshift
            modelx_manualshift = modelx_manualshift + stepsize_modelxshift
            slider.set_val(modelx_manualshift)
        elif arg_selected == 'Model Y-Shift':
            global modely_manualshift
            global stepsize_modelyshift
            modely_manualshift = modely_manualshift + stepsize_modelyshift
            slider.set_val(modely_manualshift)
        redraw_plot()
    



    #SUBTRACTOR functions to manually subtract from period, amplitude, model x-shift, and model y-shift, with given stepsize
    def subtractor(event):
        global arg_selected
        if arg_selected == 'Period':
            global per
            global stepsize_per
            temp_per = per - stepsize_per
            if temp_per > 0:
                per = temp_per
                slider.set_val(per)
        elif arg_selected == 'Amplitude':
            global amp_manual
            global stepsize_amp
            amp_manual = amp_manual - stepsize_amp
            slider.set_val(amp_manual + amp_data)
        elif arg_selected == 'Model X-Shift':
            global modelx_manualshift
            global stepsize_modelxshift
            modelx_manualshift = modelx_manualshift - stepsize_modelxshift
            slider.set_val(modelx_manualshift)
        elif arg_selected == 'Model Y-Shift':
            global modely_manualshift
            global stepsize_modelyshift
            modely_manualshift = modely_manualshift - stepsize_modelyshift
            slider.set_val(modely_manualshift)
        redraw_plot()




    #STEPSIZE functions to set stepsize size of period, amplitude, model x-shift, and model y-shift of their respective sliders
    def stepper(text):
        global arg_selected
        if evaluator(text):
            if arg_selected == 'Period':
                global stepsize_per
                stepsize_per = eval(text)
                slider.valstep = stepsize_per
            elif arg_selected == 'Amplitude':
                global stepsize_amp
                stepsize_amp = eval(text)
                slider.valstep = stepsize_amp
            elif arg_selected == 'Model X-Shift':
                global stepsize_modelxshift
                stepsize_modelxshift = eval(text)
                slider.valstep = stepsize_modelxshift
            elif arg_selected == 'Model Y-Shift':
                global stepsize_modelyshift
                stepsize_modelyshift = eval(text)
                slider.valstep = stepsize_modelyshift
            redraw_plot()




    #SET functions to manually set period, amplitude, model x-shift, and model y-shift
    def setter(text):
        global arg_selected
        if evaluator(text):
            set_num = eval(text)
            if arg_selected == 'Period':
                global per
                global slider
                if set_num > 0:
                    per = eval(text)
                    slider.set_val(per)
                else:
                    tkinter.messagebox.showwarning(title='Error', message='Period cannot be less than or equal to 0'
                                                   + f' (entered {set_num}). Old value of {per} days preserved.')
                    setbox.set_val('')
            elif arg_selected == 'Amplitude':
                global amp_data
                global amp_manual
                amp_manual = 0
                amp_data = eval(text)
                slider.set_val(amp_data)
            elif arg_selected == 'Model X-Shift':
                global modelx_manualshift
                modelx_manualshift = eval(text)
                slider.set_val(modelx_manualshift)
            elif arg_selected == 'Model Y-Shift':
                global modely_manualshift
                modely_manualshift = eval(text)
                slider.set_val(modely_manualshift)
            redraw_plot()




    #MAXXER functions to set a max period, amplitude, model x-shift, and model y-shift to their respective sliders
    def maxxer(text):
        global arg_selected
        if evaluator(text):
            slider_max = eval(text)
            if slider_max > slider.valmin:
                if arg_selected == 'Period':
                    global per_max
                    per_max = eval(text)
                    if per_max > 0:
                        slider.valmax = per_max
                        slider.ax.set_xlim(slider.valmin,per_max)
                    else:
                        tkinter.messagebox.showwarning(title='Error', message='Period cannot be less than or equal to 0'
                                                       + f' (entered {per_max}). Old value of {slider.valmax} days preserved.')
                        maxbox.set_val('')
                        return
                elif arg_selected == 'Amplitude':
                    global amp_max
                    amp_max = eval(text)
                    slider.valmax = amp_max
                    slider.ax.set_xlim(slider.valmin,amp_max)
                elif arg_selected == 'Model X-Shift':
                    global modelx_max
                    modelx_max = eval(text)
                    slider.valmax = modelx_max
                    slider.ax.set_xlim(slider.valmin,modelx_max)
                elif arg_selected == 'Model Y-Shift':
                    global modely_max
                    modely_max = eval(text)
                    slider.valmax = modely_max
                    slider.ax.set_xlim(slider.valmin,modely_max)
            else:
                tkinter.messagebox.showwarning(title='Error', message='Maximum slider value is less than or equal to minimum slider value'
                                               + f' (entered {slider_max}, min = {slider.valmin}). Old value of {slider.valmax} preserved.')
                maxbox.set_val('')
                return
            redraw_plot()
        
    def mag_maxxer(text):
        if evaluator(text):
            ymax, ymin = ax.get_ylim()
            new_ymax = eval(text)
            if new_ymax <= ymin:
                tkinter.messagebox.showwarning(title='Error', message='Maximum magnitude is less than or equal to minimum magnitude'
                                               + f' (entered {new_ymax}, min = {ymin}). Old value of {ymax} preserved.')
                magmax_box.set_val('')
                return
            else:
                ax.set_ylim(bottom = new_ymax)
            redraw_plot()




    #MINNER functions to set a minimum period, amplitude, model x-shift, and model y-shift to their respective sliders
    def minner(text):
        global arg_selected
        if evaluator(text):
            slider_min = eval(text)
            if slider_min < slider.valmax:
                if arg_selected == 'Period':
                    global per_min
                    per_min = eval(text)
                    if per_min > 0:
                        slider.valmin = per_min
                        slider.ax.set_xlim(per_min,slider.valmax)
                    else:
                        tkinter.messagebox.showwarning(title='Error', message='Period cannot be less than or equal to 0'
                                                       + f' (entered {per_min}). Old value of {slider.valmin} days preserved.')
                        minbox.set_val('')
                        return
                elif arg_selected == 'Amplitude':
                    global amp_min
                    amp_min = eval(text)
                    slider.valmin = amp_min
                    slider.ax.set_xlim(amp_min,slider.valmax)
                elif arg_selected == 'Model X-Shift':
                    global modelx_min
                    modelx_min = eval(text)
                    slider.valmin = modelx_min
                    slider.ax.set_xlim(modelx_min,slider.valmax)
                elif arg_selected == 'Model Y-Shift':
                    global modely_min
                    modely_min = eval(text)
                    slider.valmin = modely_min
                    slider.ax.set_xlim(modely_min,slider.valmax)
            else:
                tkinter.messagebox.showwarning(title='Error', message='Minimum slider value is greater than or equal to maximum slider value'
                                               + f' (entered {slider_min}, max = {slider.valmax}). Old value of {slider.valmin} preserved.')
                minbox.set_val('')
                return
            redraw_plot()
        
    def mag_minner(text):
        if evaluator(text):
            ymax, ymin = ax.get_ylim()
            new_ymin = eval(text)
            if new_ymin >= ymax:
                tkinter.messagebox.showwarning(title='Error', message='Minimum magnitude is greater than or equal to maximum magnitude'
                                               + f' (entered {new_ymin}, max = {ymax}). Old value of {ymin} preserved.')
                magmin_box.set_val('')
            else:
                ax.set_ylim(top = new_ymin)
            redraw_plot()
        
        
        

    #UPDATE functions to update period, amplitude, model x-shift, and model y-shift from their respective sliders
    def updater(val):
        global arg_selected
        if arg_selected == 'Period':
            global per
            per = slider.val
        elif arg_selected == 'Amplitude':
            global amp_manual
            amp_manual = slider.val - amp_data
        elif arg_selected == 'Model X-Shift':
            global modelx_manualshift
            modelx_manualshift = slider.val
        elif arg_selected == 'Model Y-Shift':
            global modely_manualshift
            modely_manualshift = slider.val
        redraw_plot()




    #SELECT functions change the arguments of buttons, sliders, and text boxes
    def select_period():
        Badd.label.set_text('+Period')
        Bsub.label.set_text('-Period')
        stepbox.label.set_text('Period Stepsize ')
        setbox.label.set_text('Set Period ')
        maxbox.label.set_text('Period Max ')
        minbox.label.set_text('Period Min ')
        slider.label.set_text('Period')
        slider.valmax = per_max
        slider.valmin = per_min
        slider.ax.set_xlim(per_min,per_max)
        slider.valstep = stepsize_per
        slider.set_val(per)
        redraw_plot()
    
    def select_amplitude():
        Badd.label.set_text('+Amplitude')
        Bsub.label.set_text('-Amplitude')
        stepbox.label.set_text('Amplitude Stepsize ')
        setbox.label.set_text('Set Amplitude ')
        maxbox.label.set_text('Amplitude Max ')
        minbox.label.set_text('Amplitude Min ')
        slider.label.set_text('Amplitude')
        slider.valmax = amp_max
        slider.valmin = amp_min
        slider.ax.set_xlim(amp_min,amp_max)
        slider.valstep = stepsize_amp
        slider.set_val(amp_data + amp_manual)
        redraw_plot()
    
    def select_modelyshift():
        Badd.label.set_text('+Model Y-Shift')
        Bsub.label.set_text('-Model Y-Shift')
        stepbox.label.set_text('Model Y-Shift Stepsize ')
        setbox.label.set_text('Adjust Model Y-Shift ')
        maxbox.label.set_text('Model Y-Shift Max ')
        minbox.label.set_text('Model Y-Shift Min ')
        slider.label.set_text('Model Y-Shift')
        slider.valmax = modely_max
        slider.valmin = modely_min
        slider.ax.set_xlim(modely_min,modely_max)
        slider.valstep = stepsize_modelyshift
        slider.set_val(modely_manualshift)
        redraw_plot()
    
    def select_modelxshift():
        Badd.label.set_text('+Model X-Shift')
        Bsub.label.set_text('-Model X-Shift')
        stepbox.label.set_text('Model X-Shift Stepsize ')
        setbox.label.set_text('Adjust Model X-Shift ')
        maxbox.label.set_text('Model X-Shift Max ')
        minbox.label.set_text('Model X-Shift Min ')
        slider.label.set_text('Model X-Shift')
        slider.valmax = modelx_max
        slider.valmin = modelx_min
        slider.ax.set_xlim(modelx_min,modelx_max)
        slider.valstep = stepsize_modelxshift
        slider.set_val(modelx_manualshift)
        redraw_plot()




    #create plots
    fig, ax = plt.subplots()
    plt.gca().invert_yaxis()
    plt.xlabel('Phase')
    plt.ylabel('Magnitude')
    plt.subplots_adjust(left=0.1,bottom=0.4,top=0.95)

    #gather initial data
    initial = phaser(data,per_initial,data[0][0])

    #plot initial data
    x1 = list()
    y1 = list()
    yerror = list()
    for i in initial:
        x1.append(i[0])
        y1.append(i[1])
        yerror.append(i[2])
    #p = plt.scatter(x1,y1)
    q = plt.errorbar(x1,y1,yerr = yerror,fmt="o",elinewidth=1)

    #plot initial sin curve, adjusted to data
    x_sin_curve = np.arange(0,2,0.001)
    modelx_manualshift = 0

    y_sin_datashift = 0.5 * (max(y1)+min(y1))
    modely_manualshift = 0
    y_sin_shift = y_sin_datashift

    y1_avg = sum(y1) / len(y1)
    y1_sorted = y1.copy()
    y1_sorted.sort(reverse=False)
    y1_mins = y1_sorted[0:10]
    y1_sorted.sort(reverse=True)
    y1_maxs = y1_sorted[0:10]
    y1_maxavg = sum(y1_maxs) / len(y1_maxs)
    y1_minavg = sum(y1_mins) / len(y1_mins)
    amp_manual = 0
    amp_data = round(((y1_avg - y1_minavg) + (y1_maxavg - y1_avg)) / 2,6)
    sin_width = 2 * np.pi

    y_sin_curve = np.sin(sin_width * x_sin_curve) * amp_data + y_sin_shift
    curve, = plt.plot(x_sin_curve,y_sin_curve,'g-')

    #plot initial sine scatter, adjusted to data. used for x2 calculation
    x_sin_scatter = x1
    y_sin_scatter = list()
    for x_val in x_sin_scatter:
        y_val = np.sin(sin_width * x_val) * (amp_data) + y_sin_shift
        y_sin_scatter.append(y_val)
    data_expected = plt.scatter(x_sin_scatter,y_sin_scatter,c='tab:orange')
    
    plt.legend([q, data_expected, curve], ['Observed Data with Error', 'Expected Data on Model', 'Curve Model'], loc = 'lower right')

    ymax, ymin = ax.get_ylim()
    centered = (ymax + ymin) / 2

    #calculate initial x2
    x2_calculated = find_X2(x_sin_scatter, y1, y_sin_scatter, initial)

    #create gui for textboxes, buttons, sliders
    argument_ax = plt.axes([0.06,0.05,0.08,0.12])
    model_ax = plt.axes([0.17,0.05,0.13,0.12])
    
    add = plt.axes([0.34,0.12,0.08,0.03])
    sub = plt.axes([0.34,0.07,0.08,0.03])
    steptext = plt.axes([0.66,0.12,0.05,0.03])
    settext = plt.axes([0.66,0.07,0.05,0.03])
    maxtext = plt.axes([0.51,0.12,0.05,0.03])
    mintext = plt.axes([0.51,0.07,0.05,0.03])
    magrange_ax = plt.axes([0.81,0.04,0.05,0.03])
    magmax_ax = plt.axes([0.81,0.14,0.05,0.03])
    magmin_ax = plt.axes([0.81,0.09,0.05,0.03])
    slider_ax = plt.axes([0.1,0.215,0.80,0.03])
    savepkl_ax = plt.axes([0.9,0.09,0.055,0.03])

    #make textboxes, buttons, sliders    
    argument_radio = RadioButtons(ax=argument_ax, labels=('Period', 'Amplitude', 'Model X-Shift', 'Model Y-Shift'), active=0)
    arg_selected = 'Period'
    model_radio = RadioButtons(ax=model_ax, labels=('Sine, 1 Cycle per Phase', 'Sine, 2 Cycles per Phase', 'Absolute Value Sine'), active=0)
    
    Badd = Button(ax=add,label='+Period')
    Bsub = Button(ax=sub,label='-Period')
    stepbox = TextBox(steptext,'Period Stepsize ')
    setbox = TextBox(settext,'Set Period ')
    maxbox = TextBox(maxtext,'Period Max ')
    minbox = TextBox(mintext,'Period Min ')
    magrange_box = TextBox(magrange_ax, 'Magnitude Range ')
    magmax_box = TextBox(magmax_ax, 'Magnitude Max ')
    magmin_box = TextBox(magmin_ax, 'Magnitude Min ')
    slider = Slider(slider_ax, 'Period', 0.000001, 10, valinit = 0,valstep = stepsize_per)
    slider.set_val(per)
    Bsavepkl = Button(ax=savepkl_ax,label='Save as .pkl')

    #give functions to textboxes, buttons, sliders
    argument_radio.on_clicked(arg_selector)
    model_radio.on_clicked(model_selector)
    
    Badd_id = Badd.on_clicked(adder)
    Bsub_id = Bsub.on_clicked(subtractor)
    stepbox_id = stepbox.on_submit(stepper)
    setbox_id = setbox.on_submit(setter)
    maxbox_id = maxbox.on_submit(maxxer)
    minbox_id = minbox.on_submit(minner)
    magrange_box.on_submit(mag_ranger)
    magmax_box.on_submit(mag_maxxer)
    magmin_box.on_submit(mag_minner)
    slider_id = slider.on_changed(updater)
    Bsavepkl.on_clicked(save_pkl)
    
    #create initial titles
    
    fig.suptitle('Lcvis Plot for ' + obj_name)
    
    text = fig.text(0.34,0.18,"Curve type = " + curve_type + "          "
                    + arg_selected + " Stepsize = " + str(round(slider.valstep,6)) + "          "
                    + arg_selected + " Min = " + str(round(slider.valmin,6)) + "          "
                    + arg_selected + " Max = " + str(round(slider.valmax,6)) + "          "
                    + "Plot centered at magnitude " + str(round(centered,6)))
    
    eq_info = fig.text(0.1,0.27, 'Model Equation: ' + str( round(amp_data + amp_manual,6) ) + ' * sin[' + str( round(sin_width,6) )
                        + ' * (x - ' + str( round(modelx_manualshift,6) ) + ')] + ' + str( round(y_sin_shift,6)) + '\n' 
                        + 'Model Amplitude: ' + str( round(amp_data,6) ) + '\n'
                        + 'Manually Adjusted Model X-Shift: ' + str( round(modelx_manualshift,6) ) + '\n'
                        + 'Manually Adjusted Model Y-Shift: ' + str( round(modely_manualshift,6) ) )
    
    per_info = fig.text(0.53,0.27, 'Guess Period: ' + str(period_initial) + '\n' + 'Plot Period: ' + str(round(per,6)))
    
    chi_info = fig.text(0.65,0.27, 'X2: ' + str( round(x2_calculated,6) ))
    
    ymax, ymin = ax.get_ylim()
    mag_info = fig.text(0.4,0.27, 'Magnitude Min: ' + str(round(ymin,6)) + '\n'
                        + 'Magnitude Max: ' + str(round(ymax,6)) + '\n'
                        + 'Magnitude Range: ' + str(round(ymax - ymin,6)))
    
    fig.text(0.06, 0.18, "Arguments: ")
    fig.text(0.17,0.18, "Models: ")

    per_max = 10
    per_min = 0.000001
    amp_max = round(0.5 * (max(y1)-min(y1)) + 0.1,6)
    amp_min = 0
    modelx_max = 1
    modelx_min = -1
    modely_max = max(y1)-min(y1)
    modely_min = min(y1)-max(y1)
    
    #show plot
    plt.show()

    return

import numpy as np
import argparse

#create arguments
parser = argparse.ArgumentParser()
parser.add_argument("indata")
parser.add_argument("object_name")
parser.add_argument("initial_period")
args = parser.parse_args()
indata = args.indata
object_name = args.object_name

#set initial period to 1, step size to 0.000001, initial model to sine
per = float(args.initial_period)
per_initial = per
stepsize_per = 0.000001
stepsize_phase = 0.000001
stepsize_amp = 0.000001
stepsize_modelxshift = 0.000001
stepsize_modelyshift = 0.000001
curve_type = 'sine'

#use lcvis function
use = (np.loadtxt(indata)).tolist()
name = object_name
plot = lcvis(use, name, per_initial)

# created Grant P Donnelly
# editted Jacob Juvan
