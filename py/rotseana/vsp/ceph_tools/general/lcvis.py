def lcvis( data ):

    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, TextBox, Slider

    matplotlib.use('TkAgg')

    def phaser( lightcurve, period, epoch ):

        phased = list()
        for obs in lightcurve:
            phase = (obs[0] - epoch) / period
            new = [phase % 1, obs[1], obs[2]]
            phased.append(new)

        output = list()
        for i in phased:
            output.append(i)
            tp = [i[0] + 1, i[1], i[2]]
            output.append(tp)
    
        return output

    def adder(event):
        global per
        global interval
        per = per + interval
        phased = phaser(data,per,data[0][0])
        x1 = list()
        y1 = list()
        for j in phased:
            x1.append(j[0])
            y1.append(j[1])
        d = np.vstack((x1,y1))
        p.set_offsets(d.T)
        fig.suptitle('Period: '+str(per))
        fig.canvas.draw_idle()

    def subtractor(event):
        global per
        global interval
        per = per - interval
        phased = phaser(data,per,data[0][0])
        x1 = list()
        y1 = list()
        for j in phased:
            x1.append(j[0])
            y1.append(j[1])
        d = np.vstack((x1,y1))
        p.set_offsets(d.T)
        fig.suptitle('Period: '+str(per))
        fig.canvas.draw_idle()

    def inter(text):
        global interval
        interval = eval(text)
        sper.valstep = interval

    def period(text):
        global per
        per = eval(text)
        phased = phaser(data,per,data[0][0])
        x1 = list()
        y1 = list()
        for j in phased:
            x1.append(j[0])
            y1.append(j[1])
        d = np.vstack((x1,y1))
        p.set_offsets(d.T)
        fig.suptitle('Period: '+str(per))
        fig.canvas.draw_idle()

    def maxxer(text):
        maxx = eval(text)
        sper.valmax = maxx
        sper.ax.set_xlim(0,maxx)

    def minner(text):
        minn = eval(text)
        sper.valmin = minn
        sper.ax.set_xlim(minn,sper.valmax)

    def update(val):
        global per
        per = sper.val
        phased = phaser(data,per,data[0][0])
        x1 = list()
        y1 = list()
        for j in phased:
            x1.append(j[0])
            y1.append(j[1])
        d = np.vstack((x1,y1))
        p.set_offsets(d.T)
        fig.suptitle('Period: '+str(per))
        fig.canvas.draw_idle()

    fig, ax = plt.subplots()
    fig.suptitle('Period: '+str(per))
    plt.xlabel('Phase')
    plt.ylabel('Magnitude')
    plt.subplots_adjust(left=0.1,bottom=0.3)

    initial = phaser(data,per,data[0][0])
    x = list()
    y = list()
    for i in initial:
        x.append(i[0])
        y.append(i[1])

    p = plt.scatter(x,y)

    add = plt.axes([0.25,0.1,0.1,0.1])
    sub = plt.axes([0.1,0.1,0.1,0.1])
    intertext = plt.axes([0.5,0.125,0.1,0.05])
    pertext = plt.axes([0.7,0.125,0.2,0.05])
    maxtext = plt.axes([0.85,0.025,0.05,0.05])
    mintext = plt.axes([0.1,0.025,0.05,0.05])
    axper = plt.axes([0.25,0.035,0.48,0.03])
    
    Badd = Button(ax=add,label='Add')
    Bsub = Button(ax=sub,label='Subtract')
    interbox = TextBox(intertext,'Interval')
    perbox = TextBox(pertext,'Period')
    maxbox = TextBox(maxtext,'Max')
    minbox = TextBox(mintext,'Min')
    sper = Slider(axper, 'Period', 0, 10, valinit = per,valstep = interval)
    
    Badd.on_clicked(adder)
    Bsub.on_clicked(subtractor)
    interbox.on_submit(inter)
    perbox.on_submit(period)
    maxbox.on_submit(maxxer)
    minbox.on_submit(minner)
    sper.on_changed(update)
    
    plt.show()
    
    return

import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("indata")
args = parser.parse_args()
indata = args.indata

per = 1
interval = 0.000001

use = (np.loadtxt(indata)).tolist()
plot = lcvis(use)

# Grant P Donnelly
