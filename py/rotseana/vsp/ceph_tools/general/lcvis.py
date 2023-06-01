def lcvis( data, obj_name, period_initial ):

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.widgets import Button, TextBox, Slider

    matplotlib.use('TkAgg')

    #these variables are updated throughout inside functions
    global x_sin
    global y_sin
    global x_sin_shift
    global y_sin_shift
    global y_sin_datashift
    global fitx_manualshift
    global fity_manualshift
    global amp_data
    global amp_manual
    global sin_width
    global datax_shift
    global curve_type
    global add
    global sub
    global settext
    global intertext
    global mintext
    global maxtext
    global slider_ax
    global Badd
    global Bsub
    global setbox
    global interbox
    global maxbox
    global minbox
    global slider
    global fitx_ax
    global fitx_button
    global q
    global per_max
    global per_min
    global amp_max
    global amp_min
    global datax_max
    global datax_min
    global fitx_max
    global fitx_min
    global fity_max
    global fity_min

    #PHASER function fits data to the phase plot
    def phaser( lightcurve, period, epoch ):
        phased = list()
        #this first loop adjusts the mjd of each observation to fit to the plot by subtracting the earliest date
        for obs in lightcurve:
            phase = (obs[0] - epoch + datax_shift) / period
            new = [phase % 1, obs[1], obs[2]]
            phased.append(new)
        output = list()
        #this second loop duplicates the adjusted data, shifted horizontally +1 phase
        for i in phased:
            output.append(i)
            tp = [i[0] + 1, i[1], i[2]]
            output.append(tp)
        return output

    #FIND_X2 function finds the x2 value of the data, fit to either a sine curve or parabola
    def find_X2(x_val, y_obs, y_exp, phased_data):
        #find the x2 value of the data, with expected data fitted to 1 cycle/phase sine curve
        if (curve_type == 'sine') and (sin_width == 2 * np.pi):
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
        #find the x2 value of the datas extrema, with expected data fitted to 2 cycles/phase sine curve or parabola
        if (sin_width == 4 * np.pi) or (curve_type == 'parabola'):
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
                if ((0.15 <= x_pos) and (x_pos <= .35)):
                    index = x_val.index(x_pos)
                    y_obs_l1 = y_obs[index]
                    y_exp_l1 = y_exp[index]
                    error_l1 = phased_data[index][2]
                    x_l.append(x_pos)
                    y_obs_l.append(y_obs_l1)
                    y_exp_l.append(y_exp_l1)
                    error_l.append(error_l1)
                #gather data of right minima
                if ((0.65 <= x_pos) and (x_pos <= 0.85)):
                    index = x_val.index(x_pos)
                    y_obs_r1 = y_obs[index]
                    y_exp_r1 = y_exp[index]
                    error_r1 = phased_data[index][2]
                    x_r.append(x_pos)
                    y_obs_r.append(y_obs_r1)
                    y_exp_r.append(y_exp_r1)
                    error_r.append(error_r1)
                #gather data of maximas
                if (((0 <= x_pos) and (x_pos <= 0.05))
                    or ((0.45 <= x_pos) and (x_pos <= 0.55))
                    or ((0.95 <= x_pos) and (x_pos <= 1))):
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
        #d1 = np.vstack((x1,y1))
        #p.set_offsets(d1.T)
        q.remove()
        q = ax.errorbar(x1,y1,yerr=yerror,fmt="o",c='tab:blue',elinewidth=1)

        if curve_type == 'sine':
            #update sine curve
            x_sin_curve = np.arange(0,2,0.01)
            x_sin_shift = fitx_manualshift

            y_sin_datashift = 0.5 * (max(y1)+min(y1))
            y_sin_shift = fity_manualshift + y_sin_datashift

            y_sin_curve = np.sin((sin_width * (x_sin_curve - x_sin_shift))) * (amp_data + amp_manual) + y_sin_shift
            curve.set_ydata(y_sin_curve)

            #update sine scatter
            x_sin_scatter = x1
            y_sin_scatter = list()
            for x_val in x_sin_scatter:
                y_val = np.sin((sin_width * (x_val - x_sin_shift))) * (amp_data + amp_manual) + y_sin_shift
                y_sin_scatter.append(y_val)
            d2 = np.vstack((x_sin_scatter,y_sin_scatter))
            data_expected.set_offsets(d2.T)

            #recalculate x2
            x2_calculated = find_X2(x_sin_scatter, y1, y_sin_scatter, phased)

        if curve_type == 'parabola':
            #update parabola
            x_parab = np.arange(0,2,0.01)

            y_parab_datashift = min(y1)
            y_parab_shift = y_parab_datashift + fity_manualshift

            indexes = len(x_parab)
            for index in range(indexes):
                if x_parab[index] < 0.25:
                    x_parab[index] = x_parab[index]
                if (0.25 <= x_parab[index]) and (x_parab[index] < 0.75):
                    x_parab[index] = x_parab[index] - 0.5
                if (0.75 <= x_parab[index]) and (x_parab[index] < 1.25):
                    x_parab[index] = x_parab[index] - 1.0
                if (x_parab[index] >= 1.25) and (x_parab[index] < 1.75):
                    x_parab[index] = x_parab[index] - 1.5
                if x_parab[index] >= 1.75:
                    x_parab[index] = x_parab[index] - 2.0
            y_parab = (amp_data + amp_manual) * (x_parab)**2 + y_parab_shift
            curve.set_ydata(y_parab)

            #update parabola scatter
            x_parab_data = x1
            x_parab_scatter = list()
            y_parab_scatter = list()
            indexes = len(x_parab_data)

            for x in x1:
                x_parab_scatter.append(x)

            for index in range(indexes):
                if x_parab_data[index] < 0.25:
                    x_parab_data[index] = x_parab_data[index]
                if (0.25 <= x_parab_data[index]) and (x_parab_data[index] < 0.75):
                    x_parab_data[index] = x_parab_data[index] - 0.5
                if (0.75 <= x_parab_data[index]) and (x_parab_data[index] < 1.25):
                    x_parab_data[index] = x_parab_data[index] - 1.0
                if (x_parab_data[index] >= 1.25) and (x_parab_data[index] < 1.75):
                    x_parab_data[index] = x_parab_data[index] - 1.5
                if x_parab_data[index] >= 1.75:
                    x_parab_data[index] = x_parab_data[index] - 2.0
                y_val = (amp_data + amp_manual) * (x_parab_data[index])**2 + y_parab_shift
                y_parab_scatter.append(y_val)
            d2 = np.vstack((x_parab_scatter,y_parab_scatter))
            data_expected.set_offsets(d2.T)

            #recalculate x2
            x2_calculated = find_X2(x_parab_scatter, y1, y_parab_scatter, phased)
            

        #update figure titles
        
        if curve_type == 'sine':
            if (sin_width == 2 * np.pi):
                fig.suptitle('Lcvis Plot for ' + obj_name + '\n'
                                + 'Sine Equation: ' + str( round(amp_data + amp_manual,6) ) + ' * sin[' + str( round(sin_width,6) )
                                                    + ' * (x - ' + str( round(fitx_manualshift,6) ) + ')] + ' + str( round(y_sin_shift,6)) + '                    '
                                + 'Sine Amplitude: ' + str( round(amp_data + amp_manual,6) ) + '\n'
                                + 'Manually Adjusted Data X-Shift: ' + str( round(datax_shift,6) ) + '                    '
                                + 'Manually Adjusted Fit X-Shift: ' + str( round(fitx_manualshift,6) ) + '                    '
                                + 'Manually Adjusted Fit Y-Shift: ' + str( round(fity_manualshift,6) ) + '\n'
                                + 'Unconex Period: ' + str(period_initial) + '                    ' + 'Plot Period: ' + str(per) +'\n'
                                + 'X2: ' + str( round(x2_calculated,6) ))
            if (sin_width == 4 * np.pi):
                fig.suptitle('Lcvis Plot for ' + obj_name + '\n'
                                + 'Sine Equation: ' + str( round(amp_data + amp_manual,6) ) + ' * sin[' + str( round(sin_width,6) )
                                                    + ' * (x - ' + str( round(x_sin_shift,6) ) + ')] + ' + str( round(y_sin_shift,6)) + '                    '
                                + 'Sine Amplitude: ' + str( round(amp_data + amp_manual,6) ) + '\n'
                                + 'Manually Adjusted Data X-Shift: ' + str( round(datax_shift,6) ) + '                    '
                                + 'Manually Adjusted Fit X-Shift: ' + str( round(fitx_manualshift,6) ) + '                    '
                                + 'Manually Adjusted Fit Y-Shift: ' + str( round(fity_manualshift,6) ) + '\n'
                                + 'Unconex Period: ' + str(period_initial) + '                    ' + 'Plot Period: ' + str(per) + '\n'
                                + 'Left Minima X2: ' + str( round(x2_calculated[0],6) ) + '                    '
                                + 'Right Minima X2: ' + str( round(x2_calculated[1],6) ) + '                    '
                                + 'Maxima X2: ' + str( round(x2_calculated[2],6) ))

        if curve_type == 'parabola':
            fig.suptitle('Lcvis Plot for ' + obj_name + '\n'
                            + 'Parabola Equation: ' + str( round(amp_data + amp_manual,6) ) + 'x^2 + ' + str( round(y_parab_shift,6) ) + '                    '
                            + 'Parabola Amplitude: ' + str( round(amp_data + amp_manual,6) ) + '\n'
                            + 'Manually Adjusted Data X-Shift: ' + str( round(datax_shift,6) ) + '                    '
                            + 'Manually Adjusted Fit Y-Shift: ' + str( round(fity_manualshift,6) ) + '\n'
                            + 'Unconex Period: ' + str(period_initial) + '                    ' + 'Plot Period: ' + str(per) + '\n'
                            + 'Left Minima X2: ' + str( round(x2_calculated[0],6) ) + '                    '
                            + 'Right Minima X2: ' + str( round(x2_calculated[1],6) ) + '                    '
                            + 'Maxima X2: ' + str( round(x2_calculated[2],6) ))
            
        text.set_text("Curve type = " + curve_type
                        + "       Interval = " + str(round(slider.valstep,6))
                        + "       Min = " + str(round(slider.valmin,6))
                        + "       Max = " + str(round(slider.valmax,6)))
        
        fig.canvas.draw_idle()

    #FREQUENCY functions to change the frequency of the displayed sine curve and scatter, or to change to parabola
    def frequency_1x(event):
        global sin_width
        global curve_type
        global amp_manual
        if curve_type == 'parabola':
            amp_manual = 0
        curve_type = 'sine'
        sin_width = 2 * np.pi
        reset_widgets(True, True)
        redraw_plot()

    def frequency_2x(event):
        global sin_width
        global curve_type
        global amp_manual
        if curve_type == 'parabola':
            amp_manual = 0
        curve_type = 'sine'
        sin_width = 4 * np.pi
        reset_widgets(True, True)
        redraw_plot()

    def parabola_curve(event):
        global curve_type
        global amp_manual
        if curve_type == 'sine':
            amp_manual = 5
        curve_type = 'parabola'
        reset_widgets(True, True)
        redraw_plot()

    #RESET function resets buttons, sliders, and text boxes
    def reset_widgets(insert_placeholder,check_curve):
        global add
        global sub
        global intertext
        global settext
        global maxtext
        global mintext
        global slider_ax
        global Badd
        global Bsub
        global setbox
        global interbox
        global maxbox
        global minbox
        global slider
        global fitx_ax
        global fitx_button
        global curve_type
        add.remove()
        sub.remove()
        intertext.remove()
        settext.remove()
        maxtext.remove()
        mintext.remove()
        slider_ax.remove()
        add = plt.axes([0.1,0.10,0.1,0.03])
        sub = plt.axes([0.1,0.05,0.1,0.03])
        intertext = plt.axes([0.75,0.10,0.1,0.03])
        settext = plt.axes([0.75,0.05,0.1,0.03])
        maxtext = plt.axes([0.45,0.10,0.1,0.03])
        mintext = plt.axes([0.45,0.05,0.1,0.03])
        slider_ax = plt.axes([0.1,0.15,0.80,0.03])
        if insert_placeholder:
            Badd = Button(ax=add,label='+(Choose Argument)')
            Bsub = Button(ax=sub,label='-(Choose Argument)')
            interbox = TextBox(intertext,'(Choose Argument) Interval')
            maxbox = TextBox(maxtext,'(Choose Argument) Max')
            minbox = TextBox(mintext,'(Choose Argument) Min')
            setbox = TextBox(settext,'Set (Choose Argument)')
            slider = Slider(slider_ax, '(Choose Argument)', 0, 1, valinit = 0.5, valstep = 0.000001)
        if check_curve:
            if curve_type == 'parabola':
                fitx_ax.remove()
                fitx_ax = plt.axes([0.8,0.20,0.1,0.03])
                fitx_button = Button(ax=fitx_ax,label='Unused')
            elif curve_type == 'sine':
                fitx_ax.remove()
                fitx_ax = plt.axes([0.8,0.20,0.1,0.03])
                fitx_button = Button(ax=fitx_ax,label='Fit X-Shift')
                fitx_button.on_clicked(select_fitxshift)
            
    #ADDER functions to manually add to period, phase shift, amplitude, x-shift, and y-shift, with given interval
    def adder_per(event):
        global per
        global interval_per
        per = per + interval_per
        slider.set_val(per)
        redraw_plot()

    def adder_dataxshift(event):
        global datax_shift
        global interval_phase
        datax_shift = datax_shift + interval_phase
        slider.set_val(datax_shift)
        redraw_plot()

    def adder_amp(event):
        global amp_manual
        global interval_amp
        amp_manual = amp_manual + interval_amp
        slider.set_val(amp_data + amp_manual)
        redraw_plot()

    def adder_fitxshift(event):
        global fitx_manualshift
        global interval_fitxshift
        fitx_manualshift = fitx_manualshift + interval_fitxshift
        slider.set_val(fitx_manualshift)
        redraw_plot()

    def adder_fityshift(event):
        global fity_manualshift
        global interval_fityshift
        fity_manualshift = fity_manualshift + interval_fityshift
        slider.set_val(fity_manualshift)
        redraw_plot()
    

    #SUBTRACTOR functions to manually subtract from period, data x-shift, amplitude, fit x-shift, and fit y-shift, with given interval
    def subtractor_per(event):
        global per
        global interval_per
        per = per - interval_per
        slider.set_val(per)
        redraw_plot()

    def subtractor_dataxshift(event):
        global datax_shift
        global interval_phase
        datax_shift = datax_shift - interval_phase
        slider.set_val(datax_shift)
        redraw_plot()

    def subtractor_amp(event):
        global amp_manual
        global interval_amp
        amp_manual = amp_manual - interval_amp
        slider.set_val(amp_manual + amp_data)
        redraw_plot()

    def subtractor_fitxshift(event):
        global fitx_manualshift
        global interval_fitxshift
        fitx_manualshift = fitx_manualshift - interval_fitxshift
        slider.set_val(fitx_manualshift)
        redraw_plot()

    def subtractor_fityshift(event):
        global fity_manualshift
        global interval_fityshift
        fity_manualshift = fity_manualshift - interval_fityshift
        slider.set_val(fity_manualshift)
        redraw_plot()


    #INTERVAL functions to set interval size of period, data x-shift, amplitude, fit x-shift, and fit y-shift of their respective sliders
    def inter_per(text):
        global interval_per
        interval_per = eval(text)
        slider.valstep = interval_per
        redraw_plot()

    def inter_dataxshift(text):
        global interval_phase
        interval_phase = eval(text)
        slider.valstep = interval_phase
        redraw_plot()

    def inter_amp(text):
        global interval_amp
        interval_amp = eval(text)
        slider.valstep = interval_amp
        redraw_plot()

    def inter_fitxshift(text):
        global interval_fitxshift
        interval_fitxshift = eval(text)
        slider.valstep = interval_fitxshift
        redraw_plot()

    def inter_fityshift(text):
        global interval_fityshift
        interval_fityshift = eval(text)
        slider.valstep = interval_fityshift
        redraw_plot()

    #SET functions to manually set period, data x-shift, amplitude, fit x-shift, and fit y-shift
    def set_per(text):
        global per
        global slider
        per = eval(text)
        slider.set_val(per)
        redraw_plot()

    def set_dataxshift(text):
        global datax_shift
        datax_shift = eval(text)
        slider.set_val(datax_shift)
        redraw_plot()

    def set_amp(text):
        global amp_data
        global amp_manual
        amp_manual = 0
        amp_data = eval(text)
        slider.set_val(amp_data)
        redraw_plot()

    def set_fitxshift(text):
        global fitx_manualshift
        fitx_manualshift = eval(text)
        slider.set_val(fitx_manualshift)
        redraw_plot()

    def set_fityshift(text):
        global fity_manualshift
        fity_manualshift = eval(text)
        slider.set_val(fity_manualshift)
        redraw_plot()

    #MAXXER functions to set a max period, data x-shift, amplitude, fit x-shift, and fit y-shift to their respective sliders
    def maxxer_per(text):
        global per_max
        per_max = eval(text)
        slider.valmax = per_max
        slider.ax.set_xlim(slider.valmin,per_max)
        redraw_plot()

    def maxxer_amp(text):
        global amp_max
        amp_max = eval(text)
        slider.valmax = amp_max
        slider.ax.set_xlim(slider.valmin,amp_max)
        redraw_plot()

    def maxxer_dataxshift(text):
        global datax_max
        datax_max = eval(text)
        slider.valmax = datax_max
        slider.ax.set_xlim(slider.valmin,datax_max)
        redraw_plot()

    def maxxer_fitxshift(text):
        global fitx_max
        fitx_max = eval(text)
        slider.valmax = fitx_max
        slider.ax.set_xlim(slider.valmin,fitx_max)
        redraw_plot()

    def maxxer_fityshift(text):
        global fity_max
        fity_max = eval(text)
        slider.valmax = fity_max
        slider.ax.set_xlim(slider.valmin,fity_max)
        redraw_plot()

    #MINNER functions to set a minimum period, data x-shift, amplitude, fit x-shift, and fit y-shift to their respective sliders
    def minner_per(text):
        global per_min
        per_min = eval(text)
        slider.valmin = per_min
        slider.ax.set_xlim(per_min,slider.valmax)
        redraw_plot()

    def minner_amp(text):
        global amp_min
        amp_min = eval(text)
        slider.valmin = amp_min
        slider.ax.set_xlim(amp_min,slider.valmax)
        redraw_plot()

    def minner_dataxshift(text):
        global datax_min
        datax_min = eval(text)
        slider.valmin = datax_min
        slider.ax.set_xlim(datax_min,slider.valmax)
        redraw_plot()

    def minner_fitxshift(text):
        global fitx_min
        fitx_min = eval(text)
        slider.valmin = fitx_min
        slider.ax.set_xlim(fitx_min,slider.valmax)
        redraw_plot()

    def minner_fityshift(text):
        global fity_min
        fity_min = eval(text)
        slider.valmin = fity_min
        slider.ax.set_xlim(fity_min,slider.valmax)
        redraw_plot()

    #UPDATE functions to update period, data x-shift, amplitude, fit x-shift, and fit y-shift from their respective sliders
    def update_all(val):
        global per
        per = slider.val
        redraw_plot()

    def update_amp(val):
        global amp_manual
        amp_manual = slider.val - amp_data
        redraw_plot()

    def update_dataxshift(val):
        global datax_shift
        datax_shift = slider.val
        redraw_plot()

    def update_fitxshift(val):
        global fitx_manualshift
        fitx_manualshift = slider.val
        redraw_plot()

    def update_fityshift(val):
        global fity_manualshift
        fity_manualshift = slider.val
        redraw_plot()

    #SELECT functions change the arguments of buttons, sliders, and text boxes
    def select_period(event):
        reset_widgets(False, False)
        global Badd
        global Bsub
        global setbox
        global interbox
        global maxbox
        global minbox
        global slider
        Badd = Button(ax=add,label='+Period')
        Bsub = Button(ax=sub,label='-Period')
        interbox = TextBox(intertext,'Period Interval ')
        setbox = TextBox(settext,'Set Period ')
        maxbox = TextBox(maxtext,'Period Max ')
        minbox = TextBox(mintext,'Period Min ')
        slider = Slider(slider_ax, 'Period', per_min, per_max, valinit = per,valstep = interval_per)
        Badd.on_clicked(adder_per)
        Bsub.on_clicked(subtractor_per)
        interbox.on_submit(inter_per)
        setbox.on_submit(set_per)
        maxbox.on_submit(maxxer_per)
        minbox.on_submit(minner_per)
        slider.on_changed(update_all)
        text.set_text("Curve type = " + curve_type
                        + "       Interval = " + str(round(slider.valstep,6))
                        + "       Min = " + str(round(slider.valmin,6))
                        + "       Max = " + str(round(slider.valmax,6)))
    
    def select_amplitude(event):
        reset_widgets(False, False)
        global Badd
        global Bsub
        global setbox
        global interbox
        global maxbox
        global minbox
        global slider
        global curve_type
        Badd = Button(ax=add,label='+Amplitude')
        Bsub = Button(ax=sub,label='-Amplitude')
        interbox = TextBox(intertext,'Amplitude Interval ')
        maxbox = TextBox(maxtext,'Amplitude Max ')
        minbox = TextBox(mintext,'Amplitude Min ')
        setbox = TextBox(settext,'Set Amplitude ')
        slider = Slider(slider_ax, 'Amplitude', amp_min, amp_max, valinit = amp_data + amp_manual, valstep = interval_amp)
        Badd.on_clicked(adder_amp)
        Bsub.on_clicked(subtractor_amp)
        interbox.on_submit(inter_amp)
        setbox.on_submit(set_amp)
        maxbox.on_submit(maxxer_amp)
        minbox.on_submit(minner_amp)
        slider.on_changed(update_amp)
        text.set_text("Curve type = " + curve_type
                        + "       Interval = " + str(round(slider.valstep,6))
                        + "       Min = " + str(round(slider.valmin,6))
                        + "       Max = " + str(round(slider.valmax,6)))
    
    def select_dataxshift(event):
        reset_widgets(False, False)
        global Badd
        global Bsub
        global setbox
        global interbox
        global maxbox
        global minbox
        global slider
        Badd = Button(ax=add,label='+Data X-Shift')
        Bsub = Button(ax=sub,label='-Data X-Shift')
        interbox = TextBox(intertext,'Data X-Shift Interval ')
        setbox = TextBox(settext,'Set Data X-Shift ')
        maxbox = TextBox(maxtext,'Data X-Shift Max ')
        minbox = TextBox(mintext,'Data X-Shift Min ')
        slider = Slider(slider_ax, 'Data X-Shift', datax_min, datax_max, valinit = datax_shift, valstep = interval_phase)
        Badd.on_clicked(adder_dataxshift)
        Bsub.on_clicked(subtractor_dataxshift)
        interbox.on_submit(inter_dataxshift)
        setbox.on_submit(set_dataxshift)
        maxbox.on_submit(maxxer_dataxshift)
        minbox.on_submit(minner_dataxshift)
        slider.on_changed(update_dataxshift)
        text.set_text("Curve type = " + curve_type
                        + "       Interval = " + str(round(slider.valstep,6))
                        + "       Min = " + str(round(slider.valmin,6))
                        + "       Max = " + str(round(slider.valmax,6)))
    
    def select_fityshift(event):
        reset_widgets(False, False)
        global Badd
        global Bsub
        global setbox
        global interbox
        global maxbox
        global minbox
        global slider   
        Badd = Button(ax=add,label='+Fit Y-Shift')
        Bsub = Button(ax=sub,label='-Fit Y-Shift')
        interbox = TextBox(intertext,'Fit Y-Shift Interval ')
        maxbox = TextBox(maxtext,'Fit Y-Shift Max ')
        minbox = TextBox(mintext,'Fit Y-Shift Min ')
        setbox = TextBox(settext,'Set Fit Y-Shift ')
        slider = Slider(slider_ax, 'Fit Y-Shift', fity_min, fity_max, valinit = fity_manualshift, valstep = interval_fityshift)
        Badd.on_clicked(adder_fityshift)
        Bsub.on_clicked(subtractor_fityshift)
        setbox.on_submit(set_fityshift)
        maxbox.on_submit(maxxer_fityshift)
        minbox.on_submit(minner_fityshift)
        interbox.on_submit(inter_fityshift)
        slider.on_changed(update_fityshift)
        text.set_text("Curve type = " + curve_type
                        + "       Interval = " + str(round(slider.valstep,6))
                        + "       Min = " + str(round(slider.valmin,6))
                        + "       Max = " + str(round(slider.valmax,6)))
    
    def select_fitxshift(event):
        reset_widgets(False, False)
        global Badd
        global Bsub
        global setbox
        global interbox
        global maxbox
        global minbox
        global slider  
        Badd = Button(ax=add,label='+Fit X-Shift')
        Bsub = Button(ax=sub,label='-Fit X-Shift')
        interbox = TextBox(intertext,'Fit X-Shift Interval ')
        maxbox = TextBox(maxtext,'Fit X-Shift Max ')
        minbox = TextBox(mintext,'Fit X-Shift Min ')
        setbox = TextBox(settext,'Set Fit X-Shift ')
        slider = Slider(slider_ax, 'Fit X-Shift', fitx_min, fitx_max, valinit = fitx_manualshift, valstep = interval_fitxshift)
        Badd.on_clicked(adder_fitxshift)
        Bsub.on_clicked(subtractor_fitxshift)
        setbox.on_submit(set_fitxshift)
        maxbox.on_submit(maxxer_fitxshift)
        minbox.on_submit(minner_fitxshift)
        interbox.on_submit(inter_fitxshift)
        slider.on_changed(update_fitxshift)
        text.set_text("Curve type = " + curve_type
                        + "       Interval = " + str(round(slider.valstep,6))
                        + "       Min = " + str(round(slider.valmin,6))
                        + "       Max = " + str(round(slider.valmax,6)))

    

    #create plots
    fig, ax = plt.subplots()
    plt.gca().invert_yaxis()
    plt.xlabel('Phase')
    plt.ylabel('Magnitude')
    plt.subplots_adjust(left=0.1,bottom=0.35)

    #gather initial data
    datax_shift = 0
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
    x_sin_curve = np.arange(0,2,0.01)
    fitx_manualshift = 0

    y_sin_datashift = 0.5 * (max(y1)+min(y1))
    fity_manualshift = 0
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
    amp_data = ((y1_avg - y1_minavg) + (y1_maxavg - y1_avg)) / 2
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

    #calculate initial x2
    x2_calculated = find_X2(x_sin_scatter, y1, y_sin_scatter, initial)

    #create gui for textboxes, buttons, sliders
    period_ax = plt.axes([0.1,0.20,0.1,0.03])
    amplitude_ax = plt.axes([0.275,0.20,0.1,0.03])
    datax_ax = plt.axes([0.45,0.20,0.1,0.03])
    fity_ax = plt.axes([0.625,0.20,0.1,0.03])
    fitx_ax = plt.axes([0.8,0.20,0.1,0.03])
    
    add = plt.axes([0.1,0.10,0.1,0.03])
    sub = plt.axes([0.1,0.05,0.1,0.03])
    intertext = plt.axes([0.75,0.10,0.1,0.03])
    settext = plt.axes([0.75,0.05,0.1,0.03])
    maxtext = plt.axes([0.45,0.10,0.1,0.03])
    mintext = plt.axes([0.45,0.05,0.1,0.03])
    slider_ax = plt.axes([0.1,0.15,0.80,0.03])
    
    freq_1x = plt.axes([0.45,0.25,0.1,0.03])
    freq_2x = plt.axes([0.625,0.25,0.1,0.03])
    parab = plt.axes([0.8,0.25,0.1,0.03])

    #make textboxes, buttons, sliders
    period_button = Button(ax=period_ax,label='Period')
    amplitude_button = Button(ax=amplitude_ax,label='Amplitude')
    datax_button = Button(ax=datax_ax,label='Data X-Shift')
    fity_button = Button(ax=fity_ax,label='Fit Y-Shift')
    fitx_button = Button(ax=fitx_ax,label='Fit X-Shift')
    
    Badd = Button(ax=add,label='+(Choose Argument)')
    Bsub = Button(ax=sub,label='-(Choose Argument)')
    interbox = TextBox(intertext,'(Choose Argument) Interval')
    maxbox = TextBox(maxtext,'(Choose Argument) Max')
    minbox = TextBox(mintext,'(Choose Argument) Min')
    setbox = TextBox(settext,'Set (Choose Argument)')
    slider = Slider(slider_ax, '(Choose Argument)', -1, 1, valinit = 0, valstep = 0.000001)
    
    Bfreq_1x = Button(ax=freq_1x,label='Sine, 1 Cycle per Phase')
    Bfreq_2x = Button(ax=freq_2x,label='Sine, 2 Cycles per Phase')
    Bparab = Button(ax=parab,label='Parabola')

    #give functions to textboxes, buttons, sliders
    Bfreq_1x.on_clicked(frequency_1x)
    Bfreq_2x.on_clicked(frequency_2x)
    Bparab.on_clicked(parabola_curve)
    
    period_button.on_clicked(select_period)
    amplitude_button.on_clicked(select_amplitude)
    datax_button.on_clicked(select_dataxshift)
    fity_button.on_clicked(select_fityshift)
    fitx_button.on_clicked(select_fitxshift)
    
    #create initial titles
    
    fig.suptitle('Lcvis Plot for ' + obj_name + '\n'
                    + 'Sine Equation: ' + str( round(amp_data + amp_manual,6) ) + ' * sin[' + str( round(sin_width,6) )
                                        + ' * (x - ' + str( round(fitx_manualshift,6) ) + ')] + ' + str( round(y_sin_shift,6)) + '                    '
                    + 'Sine Amplitude: ' + str( round(amp_data,6) ) + '\n'
                    + 'Manually Adjusted Data X-Shift: ' + str( round(datax_shift,6) ) + '                    '
                    + 'Manually Adjusted Fit X-Shift: ' + str( round(fitx_manualshift,6) ) + '                    '
                    + 'Manually Adjusted Fit Y-Shift: ' + str( round(fity_manualshift,6) ) + '\n'
                    + 'Unconex Period: ' + str(period_initial) + '                    ' + 'Plot Period: ' + str(per) + '\n'
                    + 'X2: ' + str( round(x2_calculated,6) ))
    
    text = fig.text(0.1,0.25,"Curve type = " + curve_type
                    + "       Interval = " + str(round(slider.valstep,6))
                    + "       Min = " + str(round(slider.valmin,6))
                    + "       Max = " + str(round(slider.valmax,6)))

    per_max = 10
    per_min = 0
    amp_max = 0.5 * (max(y1)-min(y1)) + 0.1
    amp_min = 0
    datax_max = 1
    datax_min = -1
    fitx_max = 1
    fitx_min = -1
    fity_max = max(y1)-min(y1)
    fity_min = min(y1)-max(y1)

    plt.legend([q, data_expected, curve], ['Observed Data with Error', 'Expected Data on Fit', 'Curve Fit'], bbox_to_anchor=(1,6.2))
    
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

#set initial period to 1, step size to 0.000001, initial fit to sine
per = float(args.initial_period)
per_initial = per
interval_per = 0.000001
interval_phase = 0.000001
interval_amp = 0.000001
interval_fitxshift = 0.000001
interval_fityshift = 0.000001
curve_type = 'sine'

#use lcvis function
use = (np.loadtxt(indata)).tolist()
name = object_name
plot = lcvis(use, name, per_initial)

# created Grant P Donnelly
# editted Jacob Juvan
