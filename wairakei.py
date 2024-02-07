##########################################################################
##########################################################################
##
##                What are you doing looking at this file?
##
##########################################################################
##########################################################################
#
# Just kidding. There is some useful stuff in here that will help you complete
# some of the labs and your project. Feel free to adapt it. 
#
# (Sorry about the awful commenting though. Do as I say, not as I do, etc...)

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from ipywidgets import interact, fixed, interactive_output, HBox, Button, VBox, Output, IntSlider, Checkbox, FloatSlider, FloatLogSlider, Dropdown
TEXTSIZE = 16
from IPython.display import clear_output
import time
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm as colmap
from copy import copy
from scipy.stats import multivariate_normal

# commenting in here is pretty shocking tbh

# wairakei model
def	wairakei_data():
	# load some data
	tq, q = np.genfromtxt('wk_production_history.csv', delimiter=',', unpack=True)
	tp, p = np.genfromtxt('wk_pressure_history.csv', delimiter=',', unpack=True)

	# plot some data
	f,ax1 = plt.subplots(1,1,figsize=(12,6))
	ax1.plot(tq,q,'b-',label='production')
	ax1.plot([],[],'ro',label='pressure')
	ax1.set_xlabel('time [yr]',size=TEXTSIZE)
	ax1.set_ylabel('production rate [kg/s]',size=TEXTSIZE)
	
	ax2 = ax1.twinx()
	ax2.plot(tp,p,'ro')
	v = 2.
	for tpi,pi in zip(tp,p):
		ax2.plot([tpi,tpi],[pi-v,pi+v], 'r-', lw=0.5)
	ax2.set_ylabel('pressure [bar]',size=TEXTSIZE);
	for ax in [ax1,ax2]: 
		ax.tick_params(axis='both',labelsize=TEXTSIZE)
		ax.set_xlim([None,1980])
	ax1.legend(prop={'size':TEXTSIZE})
	plt.show()
def lpm_plot(i=1):
	f,ax = plt.subplots(1,1, figsize=(12,6))
	ax.axis('off')
	
	ax.set_xlim([0,1])
	ax.set_ylim([0,1])
	
	r = 0.3
	cx,cy = [0.5,0.35]
	h = 0.3
	dh = -0.13
	dh2 = 0.05
	e = 4.
	
	th = np.linspace(0,np.pi,101)
	col = 'r'
	
	ax.fill_between([0,1],[0,0],[1,1],color='b',alpha=0.1, zorder = 0)
		
	ax.plot(cx + r*np.cos(th), cy + r*np.sin(th)/e, color = col, ls = '-')
	ax.plot(cx + r*np.cos(th), cy - r*np.sin(th)/e, color = col, ls = '-')
	ax.plot(cx + r*np.cos(th), cy + r*np.sin(th)/e+h, color = col, ls = '--')
	ax.plot(cx + r*np.cos(th), cy - r*np.sin(th)/e+h, color = col, ls = '--')
	ax.plot([cx+r,cx+r],[cy,cy+h],color=col,ls='--')
	ax.plot([cx-r,cx-r],[cy,cy+h],color=col,ls='--')
	
	ax.plot(cx + r*np.cos(th), cy + r*np.sin(th)/e+h+(i>0)*dh+(i>1)*dh2, color = col, ls = '-')
	ax.plot(cx + r*np.cos(th), cy - r*np.sin(th)/e+h+(i>0)*dh+(i>1)*dh2, color = col, ls = '-')
		
	ax.plot([cx+r,cx+r],[cy,cy+h+(i>0)*dh+(i>1)*dh2],color=col,ls='-')
	ax.plot([cx-r,cx-r],[cy,cy+h+(i>0)*dh+(i>1)*dh2],color=col,ls='-')
		
	ax.fill_between(cx + r*np.cos(th),cy - r*np.sin(th)/e,cy + r*np.sin(th)/e+h+(i>0)*dh+(i>1)*dh2, color='r', alpha = 0.1)
	
	if i > 0:
		cube(ax, 0.90, 0.8, 0.025, 'r')
		ax.arrow(cx+1.05*r,cy+1.2*(h+dh)+0.05, 0.05, 0.14, color = 'r', head_width=0.02, head_length=0.04, length_includes_head=True)
		
	if i > 1:
		cube(ax, 0.85, 0.5, 0.015, 'b')
		cube(ax, 0.15, 0.5, 0.015, 'b')
		cube(ax, 0.85, 0.35, 0.015, 'b')
		cube(ax, 0.15, 0.35, 0.015, 'b')
		cube(ax, 0.25, 0.23, 0.015, 'b')
		cube(ax, 0.50, 0.18, 0.015, 'b')
		cube(ax, 0.75, 0.23, 0.015, 'b')
		
		ax.arrow(0.17,0.5,0.02,0.0, color = 'b', head_width=0.02, head_length=0.01, length_includes_head=True)
		ax.arrow(0.83,0.5,-0.02,0.0, color = 'b', head_width=0.02, head_length=0.01, length_includes_head=True)
		ax.arrow(0.17,0.35,0.02,0.0, color = 'b', head_width=0.02, head_length=0.01, length_includes_head=True)
		ax.arrow(0.83,0.35,-0.02,0.0, color = 'b', head_width=0.02, head_length=0.01, length_includes_head=True)
		ax.arrow(0.50,0.21,0.,0.04, color = 'b', head_width=0.01, head_length=0.02, length_includes_head=True)
		ax.arrow(0.26,0.25,0.015,0.025, color = 'b', head_width=0.015, head_length=0.01, length_includes_head=True)
		ax.arrow(0.74,0.25,-0.015,0.025, color = 'b', head_width=0.015, head_length=0.01, length_includes_head=True)
		
	if i > 2:
		for fr in [0.35,0.70,0.90]:
			ax.plot(cx + r*np.cos(th), cy + r*np.sin(th)/e+h+fr*(dh+dh2), color = 'k', ls = '--')
			ax.plot(cx + r*np.cos(th), cy - r*np.sin(th)/e+h+fr*(dh+dh2), color = 'k', ls = '--')
			
			ax.fill_between(cx + r*np.cos(th), cy - r*np.sin(th)/e+h+fr*(dh+dh2), cy + r*np.sin(th)/e+h+fr*(dh+dh2), color = 'k', alpha = 0.1)
			
			ax.arrow(0.18, cy+h, 0, dh+dh2, color = 'k', head_width=0.01, head_length=0.02, length_includes_head=True)
			ax.text(0.17, cy+h+0.5*(dh+dh2), 'lowers\nover time', color='k', ha = 'right', va='center', size=TEXTSIZE-1, fontstyle = 'italic')
		
	xt1,xt2,xt3,xt4 = [0.2,0.06,0.07,0.07]
	yt = 0.85
	yt2 = 0.05
	ax.text(xt1,yt,r'$\dot{P}$ =', color = 'k', size = TEXTSIZE+4)
	if i == 0:
		ax.text(xt1+xt2,yt,r'$0$', color = 'k', size = TEXTSIZE+4)
	if i > 0:
		ax.text(xt1+xt2,yt,r'$-aq$', color = 'r', size = TEXTSIZE+4)
	if i > 1:
		ax.text(xt1+xt2+xt3,yt,r'$-bP$', color = 'b', size = TEXTSIZE+4)
	if i > 2:
		ax.text(xt1+xt2+xt3+xt4,yt,r'$-c\dot{q}$', color = 'k', size = TEXTSIZE+4)
		
	if i == 0:
		ax.text(0.5, yt2, 'reservoir initially at pressure equilibrium', size = TEXTSIZE+4, ha = 'center', va = 'bottom', fontstyle = 'italic')
	elif i == 1:
		ax.text(0.5, yt2, 'extraction from reservoir at rate, $q$', size = TEXTSIZE+4, ha = 'center', va = 'bottom', fontstyle = 'italic')
	elif i == 2:
		ax.text(0.5, yt2, 'recharge from surrounding rock, proportional to $P$', size = TEXTSIZE+4, ha = 'center', va = 'bottom', fontstyle = 'italic')
	elif i == 3:
		ax.text(0.5, yt2, 'response to extraction not instantaneous: "slow drainage", $\dot{q}$', size = TEXTSIZE+4, ha = 'center', va = 'bottom', fontstyle = 'italic')
	
	plt.show()
def cube(ax,x0,y0,dx,col):	
	dy = dx*2.
	s2 = 2
	ax.plot([x0+dx/s2,x0, x0-dx,x0-dx,x0,x0],[y0+dy/s2,y0,y0,y0-dy,y0-dy,y0],color=col,ls='-')
	ax.plot([x0-dx,x0-dx+dx/s2,x0+dx/s2,x0+dx/s2,x0],[y0,y0+dy/s2,y0+dy/s2,y0+dy/s2-dy,y0-dy],color=col,ls='-')
	ax.fill_between([x0-dx,x0-dx+dx/s2,x0,x0+dx/s2],[y0-dy,y0-dy,y0-dy,y0-dy+dy/s2],[y0,y0+dy/s2,y0+dy/s2,y0+dy/s2],color=col,alpha=0.1)
def lpm_demo():
	sldr = IntSlider(value=0, description='slide me!', min = 0, max = 3, step = 1, continuous_update = False, readout=False)
	return VBox([sldr, interactive_output(lpm_plot, {'i':sldr})])
def plot_lpm_models(a,b,c):
	# load some data
	tq,q = np.genfromtxt('wk_production_history.csv', delimiter = ',')[:28,:].T
	tp,p = np.genfromtxt('wk_pressure_history.csv', delimiter = ',')[:28,:].T
	dqdt = 0.*q                 # allocate derivative vector
	dqdt[1:-1] = (q[2:]-q[:-2])/(tq[2:]-tq[:-2])    # central differences
	dqdt[0] = (q[1]-q[0])/(tq[1]-tq[0])             # forward difference
	dqdt[-1] = (q[-1]-q[-2])/(tq[-1]-tq[-2])        # backward difference
	
	# plot the data with error bars
	f,ax = plt.subplots(1,1,figsize=(12,6))
	ax.set_xlabel('time [yr]',size=TEXTSIZE)
	ax.plot(tp,p,'ro', label = 'observations')
	v = 2.
	for tpi,pi in zip(tp,p):
		ax.plot([tpi,tpi],[pi-v,pi+v], 'r-', lw=0.5)
		
	# define derivative function
	def lpm(pi,t,a,b,c):                 # order of variables important
		qi = np.interp(t,tq,q)           # interpolate (piecewise linear) flow rate
		dqdti = np.interp(t,tq,dqdt)     # interpolate derivative
		return -a*qi - b*pi - c*dqdti    # compute derivative

	# implement an improved Euler step to solve the ODE
	def solve_lpm(t,a,b,c):
		pm = [p[0],]                            # initial value
		for t0,t1 in zip(tp[:-1],tp[1:]):           # solve at pressure steps
			dpdt1 = lpm(pm[-1]-p[0], t0, a, b, c)   # predictor gradient
			pp = pm[-1] + dpdt1*(t1-t0)             # predictor step
			dpdt2 = lpm(pp-p[0], t1, a, b, c)       # corrector gradient
			pm.append(pm[-1] + 0.5*(t1-t0)*(dpdt2+dpdt1))  # corrector step
		return np.interp(t, tp, pm)             # interp onto requested times

	# solve and plot model
	pm = solve_lpm(tp,a,b,c)
	ax.plot(tp, pm, 'k-', label='model')
	
	# axes upkeep
	ax.set_ylabel('pressure [bar]',size=TEXTSIZE);
	ax.tick_params(axis='both',labelsize=TEXTSIZE)
	ax.legend(prop={'size':TEXTSIZE})
	plt.show()
def lpm_model():
    # load flow rate data and compute derivative
    tq,q = np.genfromtxt('wk_production_history.csv', delimiter = ',')[:28,:].T
    tp,p = np.genfromtxt('wk_pressure_history.csv', delimiter = ',')[:28,:].T

    dqdt = 0.*q                 # allocate derivative vector
    dqdt[1:-1] = (q[2:]-q[:-2])/(tq[2:]-tq[:-2])    # central differences
    dqdt[0] = (q[1]-q[0])/(tq[1]-tq[0])             # forward difference
    dqdt[-1] = (q[-1]-q[-2])/(tq[-1]-tq[-2])        # backward difference

    # define derivative function
    def lpm(pi,t,a,b,c):                 # order of variables important
        qi = np.interp(t,tq,q)           # interpolate (piecewise linear) flow rate
        dqdti = np.interp(t,tq,dqdt)     # interpolate derivative
        return -a*qi - b*pi - c*dqdti    # compute derivative

    # implement an imporved Euler step to solve the ODE
    def solve_lpm(t,a,b,c):
        pm = [p[0],]                            # initial value
        for t0,t1 in zip(tp[:-1],tp[1:]):           # solve at pressure steps
            dpdt1 = lpm(pm[-1]-p[0], t0, a, b, c)   # predictor gradient
            pp = pm[-1] + dpdt1*(t1-t0)             # predictor step
            dpdt2 = lpm(pp-p[0], t1, a, b, c)       # corrector gradient
            pm.append(pm[-1] + 0.5*(t1-t0)*(dpdt2+dpdt1))  # corrector step
        return np.interp(t, tp, pm)             # interp onto requested times
        
    # use CURVE_FIT to find "best" model
    from scipy.optimize import curve_fit
    pars = curve_fit(solve_lpm, tp, p, [1,1,1])[0]

    # plot the best solution
    pm = solve_lpm(tp,*pars)
    f,ax = plt.subplots(1,1,figsize=(12,6))
    ax.plot(tp, p, 'ro', label = 'observations')
    ax.plot(tp, pm, 'k-', label='model')
    ax.set_ylabel("pressure [bar]",size=14); ax.set_xlabel("time",size=14)
    ax.legend(prop={'size':14})
    ax.set_ylim([25,60])
    ax.set_title('a={:2.1e},   b={:2.1e},   c={:2.1e}'.format(*pars),size=14);
def lpm_models():
	a0,b0,c0 = [2.2e-3,1.1e-1,6.8e-3]
	dlog = 0.1
	a = FloatLogSlider(value=a0, base=10, description=r'$a$', min = np.log10(a0)-dlog, max = np.log10(a0)+dlog, step = dlog/10, continuous_update = False)
	b = FloatLogSlider(value=b0, base=10, description=r'$b$', min = np.log10(b0)-dlog, max = np.log10(b0)+dlog, step = dlog/10, continuous_update = False)
	dlog*=5
	c = FloatLogSlider(value=c0, base=10, description=r'$c$', min = np.log10(c0)-dlog, max = np.log10(c0)+dlog, step = dlog/10, continuous_update = False)
	io = interactive_output(plot_lpm_models, {'a':a,'b':b,'c':c})
	return VBox([HBox([a,b,c]),io])
def plot_lpm_posterior(sa,sb,sc,Nmods):
	# load some data
	tq, q = np.genfromtxt('wk_production_history.csv', delimiter=',', unpack=True)
	tp, p = np.genfromtxt('wk_pressure_history.csv', delimiter=',', unpack=True)
	dqdt = 0.*q                 # allocate derivative vector
	dqdt[1:-1] = (q[2:]-q[:-2])/(tq[2:]-tq[:-2])    # central differences
	dqdt[0] = (q[1]-q[0])/(tq[1]-tq[0])             # forward difference
	dqdt[-1] = (q[-1]-q[-2])/(tq[-1]-tq[-2])        # backward difference
	
	a0,b0,c0 = [2.2e-3,1.1e-1,6.8e-3]
	dlog = 0.1
	Nmods = int(Nmods)
	
	a = np.random.randn(Nmods)*sa+a0
	b = np.random.randn(Nmods)*sb+b0
	c = np.random.randn(Nmods)*sc+c0
	
	# plot the data with error bars
	f = plt.figure(figsize=(12,6))
	ax = plt.axes([0.15,0.15,0.5,0.7])
	ax1 = plt.axes([0.70,0.69,0.2,0.15])
	ax2 = plt.axes([0.70,0.42,0.2,0.15])
	ax3 = plt.axes([0.70,0.15,0.2,0.15])
	for m0,sm,axi,mv in zip([a0,b0,c0],[sa,sb,sc],[ax1,ax2,ax3],[a,b,c]): 
		axi.set_yticks([])
		if sm < 1.e-6:
			axi.plot([m0-3*dlog*m0, m0,m0,m0,m0+3*dlog*m0],[0,0,1,0,0],'r-',zorder=2)
		else:
			x = np.linspace(m0-3*dlog*m0, m0+3*dlog*m0, 101)
			y = np.exp(-(x-m0)**2/(2*sm**2))/np.sqrt(2*np.pi*sm**2)
			axi.plot(x,y,'r-',zorder=2)
		
		bins = np.linspace(m0-3*dlog*m0, m0+3*dlog*m0, int(4*np.sqrt(Nmods))+1)
		h,e = np.histogram(mv, bins)
		h = h/(np.sum(h)*(e[1]-e[0]))
		axi.bar(e[:-1],h,e[1]-e[0], color = [0.5,0.5,0.5])
		
		if axi is ax2: dlog*=5
	ax1.set_xlabel('$a$',size=TEXTSIZE)
	ax2.set_xlabel('$b$',size=TEXTSIZE)
	ax3.set_xlabel('$c$',size=TEXTSIZE)
	
	ax.set_xlabel('time [yr]',size=TEXTSIZE)
	ax.plot(tp,p,'ro', label = 'observations')
	v = 2.
	for tpi,pi in zip(tp,p):
		ax.plot([tpi,tpi],[pi-v,pi+v], 'r-', lw=0.5)
		
	# define derivative function
	def lpm(pi,t,a,b,c):                 # order of variables important
		qi = np.interp(t,tq,q)           # interpolate (piecewise linear) flow rate
		dqdti = np.interp(t,tq,dqdt)     # interpolate derivative
		return -a*qi - b*pi - c*dqdti    # compute derivative

	# implement an improved Euler step to solve the ODE
	def solve_lpm(t,a,b,c):
		pm = [p[0],]                            # initial value
		for t0,t1 in zip(tp[:-1],tp[1:]):           # solve at pressure steps
			dpdt1 = lpm(pm[-1]-p[0], t0, a, b, c)   # predictor gradient
			pp = pm[-1] + dpdt1*(t1-t0)             # predictor step
			dpdt2 = lpm(pp-p[0], t1, a, b, c)       # corrector gradient
			pm.append(pm[-1] + 0.5*(t1-t0)*(dpdt2+dpdt1))  # corrector step
		return np.interp(t, tp, pm)             # interp onto requested times

	# solve and plot model
	alpha = np.min([0.5,10./Nmods])
	lw = 0.5
	for ai,bi,ci in zip(a,b,c):
		pm = solve_lpm(tp,ai,bi,ci)
		ax.plot(tp, pm, 'k-', alpha = alpha, lw = lw)
	ax.plot([],[],'k-',alpha=alpha,lw=lw,label='possible models')
	
	# axes upkeep
	pm = solve_lpm(tp,a0,b0,c0)
	ax.plot(tp, pm, 'k-', lw = 2, label = 'best model')
	ax.set_ylabel('pressure [bar]',size=TEXTSIZE);
	ax.tick_params(axis='both',labelsize=TEXTSIZE)
	ax.legend(prop={'size':TEXTSIZE})
	ax.set_xlim([None,1980])
	ax.set_title(r'$\sigma_a='+'{:2.1e}'.format(sa)+r'$,   $\sigma_b='+'{:2.1e}'.format(sb)+r'$,   $\sigma_c='+'{:2.1e}'.format(sc)+'$',size=TEXTSIZE);
	plt.show()
def lpm_posterior():
	a0,b0,c0 = [2.2e-3,1.1e-1,6.8e-3]
	dlog = 0.1
	sa = FloatSlider(value=dlog*a0/2, description=r'$\sigma_a$', min = 0., max = dlog*a0, step = dlog*a0/10., continuous_update = False)
	sb = FloatSlider(value=dlog*b0/2, description=r'$\sigma_b$', min = 0., max = dlog*b0, step = dlog*b0/10., continuous_update = False)
	dlog*=5
	sc = FloatSlider(value=dlog*c0/2, description=r'$\sigma_c$', min = 0., max = dlog*c0, step = dlog*c0/10., continuous_update = False)
	Nmods = FloatLogSlider(value = 4, base=2, description='samples', min = 0, max = 8, step = 1, continuous_update=False)
	io = interactive_output(plot_lpm_posterior, {'sa':sa,'sb':sb,'sc':sc,'Nmods':Nmods})
	return VBox([HBox([sa,sb,sc,Nmods]),io])
def plot_lpm_prediction(Nmods, reveal, sa, sb, sc):
	# load some data
	tq, q = np.genfromtxt('wk_production_history.csv', delimiter=',', unpack=True)
	tp, p = np.genfromtxt('wk_pressure_history.csv', delimiter=',', unpack=True)
	dqdt = 0.*q                 # allocate derivative vector
	dqdt[1:-1] = (q[2:]-q[:-2])/(tq[2:]-tq[:-2])    # central differences
	dqdt[0] = (q[1]-q[0])/(tq[1]-tq[0])             # forward difference
	dqdt[-1] = (q[-1]-q[-2])/(tq[-1]-tq[-2])        # backward difference
	
	if not reveal:
		iq = np.argmin(abs(tq-1981))
		ip = np.argmin(abs(tp-1981))
	else:
		iq = len(tq)
		ip = len(tp)
	
	a0,b0,c0 = [2.2e-3,1.1e-1,6.8e-3]
	dlog = 0.1
	Nmods = int(Nmods)
	
	np.random.seed(13)
	a = np.random.randn(Nmods)*sa+a0
	b = np.random.randn(Nmods)*sb+b0
	c = np.random.randn(Nmods)*sc+c0
	
	# plot the data with error bars
	f = plt.figure(figsize=(15,5))
	ax = plt.axes([0.15,0.15,0.5,0.7])
	ax2 = plt.axes([0.75,0.15,0.20,0.7])
	ax.set_xlabel('time [yr]',size=TEXTSIZE)
	ax.plot(tp[:ip],p[:ip],'ro', label = 'observations')
	
	v = 2.
	for tpi,pi in zip(tp[:ip],p[:ip]):
		ax.plot([tpi,tpi],[pi-v,pi+v], 'r-', lw=0.5)
		
	# define derivative function
	def lpm(pi,t,a,b,c):                 # order of variables important
		qi = np.interp(t,tq,q)           # interpolate (piecewise linear) flow rate
		dqdti = np.interp(t,tq,dqdt)     # interpolate derivative
		return -a*qi - b*pi - c*dqdti    # compute derivative

	# implement an improved Euler step to solve the ODE
	def solve_lpm(t,a,b,c):
		pm = [p[0],]                            # initial value
		for t0,t1 in zip(tp[:-1],tp[1:]):           # solve at pressure steps
			dpdt1 = lpm(pm[-1]-p[0], t0, a, b, c)   # predictor gradient
			pp = pm[-1] + dpdt1*(t1-t0)             # predictor step
			dpdt2 = lpm(pp-p[0], t1, a, b, c)       # corrector gradient
			pm.append(pm[-1] + 0.5*(t1-t0)*(dpdt2+dpdt1))  # corrector step
		return np.interp(t, tp, pm)             # interp onto requested times

	# solve and plot model
	alpha = np.min([0.5,10./Nmods])
	lw = 0.5
	pmf = []
	for ai,bi,ci in zip(a,b,c):
		pm = solve_lpm(tp,ai,bi,ci)
		ax.plot(tp, pm, 'k-', alpha = alpha, lw = lw)
		pmf.append(pm[-1])
	ax.plot([],[],'k-',alpha=0.5,lw=lw,label='possible models')
	pm = solve_lpm(tp,a0,b0,c0)
	ax.plot(tp, pm, 'k-', lw = 2, label = 'best model')
	ax.axvline(tp[-1], color = 'k', linestyle = ':', label='predict future')
	
	bins = np.linspace(np.min(pmf)*0.999, np.max(pmf)*1.001, int(np.sqrt(Nmods))+1)
	h,e = np.histogram(pmf, bins)
	h = h/(np.sum(h)*(e[1]-e[0]))
	ax2.bar(e[:-1],h,e[1]-e[0], color = [0.5,0.5,0.5])
	ax2.set_xlim([30,45])
	ax2.set_ylim([0,1])
	
	if Nmods>10:
		ax2.axvline(pm[-1], label='best model',color = 'k', linestyle = '-')
		if reveal:
			ax2.axvline(p[-1], label='true process',color = 'r', linestyle = '-')
			ax2.fill_between([p[-1]-v, p[-1]+v], [0,0], [1,1], color='r', alpha=0.5)
		
		yf5,yf95 = np.percentile(pmf, [5,95])
		ax2.axvline(yf5, label='90% interval',color = 'k', linestyle = '--')
		ax2.axvline(yf95, color = 'k', linestyle = '--')
	
	# axes upkeep
	ax.set_ylabel('pressure [bar]',size=TEXTSIZE);
	ax2.set_xlabel('pressure [bar]',size=TEXTSIZE);
	ax2.set_ylabel('probability',size=TEXTSIZE)
	for axi in [ax,ax2]: axi.tick_params(axis='both',labelsize=TEXTSIZE)
	ax.legend(prop={'size':TEXTSIZE})
	plt.show()
def lpm_prediction(sa,sb,sc):	
	Nmods = FloatLogSlider(value = 64, base=4, description='samples', min = 0, max = 5, step = 1, continuous_update=False)
	reveal = Checkbox(value = False, description='reveal future!')
	io = interactive_output(plot_lpm_prediction, {'Nmods':Nmods, 'reveal':reveal, 'sa':fixed(sa), 'sb':fixed(sb), 'sc':fixed(sc)})
	return VBox([HBox([Nmods, reveal]),io])
def plot_lpm_structural(c,reveal):
	# load some data
	tq, q = np.genfromtxt('wk_production_history.csv', delimiter=',', unpack=True)
	tp, p = np.genfromtxt('wk_pressure_history.csv', delimiter=',', unpack=True)
	dqdt = 0.*q                 # allocate derivative vector
	dqdt[1:-1] = (q[2:]-q[:-2])/(tq[2:]-tq[:-2])    # central differences
	dqdt[0] = (q[1]-q[0])/(tq[1]-tq[0])             # forward difference
	dqdt[-1] = (q[-1]-q[-2])/(tq[-1]-tq[-2])        # backward difference
	
	if not reveal:
		iq = np.argmin(abs(tq-1981))
		ip = np.argmin(abs(tp-1981))
	else:
		iq = len(tq)
		ip = len(tp)
		
	# define derivative function
	def lpm(pi,t,a,b,c):                 # order of variables important
		qi = np.interp(t,tq,q)           # interpolate (piecewise linear) flow rate
		dqdti = np.interp(t,tq,dqdt)     # interpolate derivative
		return -a*qi - b*pi - c*dqdti    # compute derivative

	# implement an improved Euler step to solve the ODE
	def solve_lpm(t,a,b,c):
		pm = [p[0],]                            # initial value
		for t0,t1 in zip(tp[:-1],tp[1:]):           # solve at pressure steps
			dpdt1 = lpm(pm[-1]-p[0], t0, a, b, c)   # predictor gradient
			pp = pm[-1] + dpdt1*(t1-t0)             # predictor step
			dpdt2 = lpm(pp-p[0], t1, a, b, c)       # corrector gradient
			pm.append(pm[-1] + 0.5*(t1-t0)*(dpdt2+dpdt1))  # corrector step
		return np.interp(t, tp, pm)             # interp onto requested times
		
	solve_lpm_c0 = lambda t,a,b: solve_lpm(t,a,b,c)
	a,b = curve_fit(solve_lpm_c0, tp[:28], p[:28], [1,1])[0]
	
	#a0,b0 = [4.72e-3,2.64e-1]
	#dlog = 0.1
	#Nmods = 64
	
	#np.random.seed(13)
	#a = np.random.randn(Nmods)*sa+a0
	#b = np.random.randn(Nmods)*sb+b0
	
	# plot the data with error bars
	f = plt.figure(figsize=(15,5))
	ax = plt.axes([0.1,0.15,0.8,0.7])
	
	ax.set_xlabel('time [yr]',size=TEXTSIZE)
	ax.plot(tp[:ip],p[:ip],'ro', label = 'observations')
	v = 2.
	for tpi,pi in zip(tp[:ip],p[:ip]):
		ax.plot([tpi,tpi],[pi-v,pi+v], 'r-', lw=0.5)
	
	# solve and plot model
	#alpha = np.min([0.5,10./Nmods])
	#lw = 0.5
	#for ai,bi in zip(a,b):
	#	pm = solve_lpm(tp[:ip],ai,bi,c)
	#	ax.plot(tp[:ip], pm, 'k-', alpha = alpha, lw = lw)
	#ax.plot([],[],'k-',alpha=alpha,lw=lw,label='possible models')
	
	# axes upkeep
	pm = solve_lpm(tp[:ip],a,b,c)
	ax.plot(tp[:ip], pm, 'k-', lw = 2, label = 'best model')
	ax.set_ylabel('pressure [bar]',size=TEXTSIZE);
	ax.tick_params(axis='both',labelsize=TEXTSIZE)
	ax.legend(prop={'size':TEXTSIZE})
	ax.set_xlim([1952,2012])
	ax.set_ylim([25,60])	
	ax.set_title(r'$a='+'{:2.1e}'.format(a)+r'$,   $b='+'{:2.1e}'.format(b)+r'$,   $c='+'{:2.1e}'.format(c)+'$',size=TEXTSIZE);
	plt.show()
def lpm_structural():
	c = FloatSlider(value=0, description=r'$c$', min = 0., max = 1.2e-2, step = 1.e-3, continuous_update = False)
	#dlog*=5
	reveal = Checkbox(value = False, description='reveal future!')
	io = interactive_output(plot_lpm_structural, {'c':c,'reveal':reveal})
	return VBox([HBox([c,reveal]),io])
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	