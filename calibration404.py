import numpy as np
from matplotlib import pyplot as plt
from ipywidgets import interactive_output, HBox, Button, VBox, Output, IntSlider, Checkbox, FloatSlider
TEXTSIZE = 16
from IPython.display import clear_output
import time
from matplotlib import cm as colmap

# One day I'll write proper comments in here. But not today. Venture on at your own risk.

# manufacture suspension data
def suspension(t,c,k,m,x0,dxdt0):
	# implements the suspension model (uses complex varaibles but returns real displacements)
	c += 0j
	k += 0j
	rt1,rt2 = -c/(2*m)+np.array([-1.,1.])*np.sqrt(c**2/m**2-4*k/m)
	
	x0 += 0j
	dxdt0 += 0j
	A = (dxdt0 - x0*rt2)/(rt1-rt2)
	B = x0 - A
	
	return np.real(A*np.exp(rt1*t) + B*np.exp(rt2*t))
def get_suspension_data(noisy=False):
	
	# 'true' parameters
	m = 1000.
	c = 24.
	k = 9.
	x0 = 1
	dxdt0 = 0.1
	
	# observation times
	t = np.linspace(0,100,33)
	x = suspension(t,c,k,m,x0,dxdt0)
	
	# add noise if requested
	if noisy:
		# set seed so get same random obs
		np.random.seed(1)
		t1,t2 = [20,60]
		inds = np.where((t>t1)&(t<t2))
		x[inds] = x[inds] + 3.*(np.random.rand(len(inds[0]))-0.5)
	
	return t,x,(m,x0,dxdt0)
# plot data and car suspension model
def plot_suspension_model():
	f = plt.figure(figsize=(12,6))
	ax = plt.axes([0.1,0.1,0.8,0.8])
	
	t,x,pars = get_suspension_data()
	m,x0,dxdt0 = pars
	
	ax.plot(t,np.real(x),'bo',mfc='w',mew=1.5,label='data')
	
	for t in ax.get_xticklabels()+ax.get_yticklabels():
		t.set_fontsize(TEXTSIZE)
		
	ax.set_xlabel('time [s]',size=TEXTSIZE)
	ax.set_ylabel('displacement [m]', size=TEXTSIZE)
	
	np.random.seed(int(time.time()))
	cm = np.random.rand()*100
	km = np.random.rand()*25+2
	t = np.linspace(0,100,201)
	x = suspension(t,cm,km,*pars)
	
	ax.plot(t,x,'k-',mfc='w',mew=1.5,label='model')
	
	ax.legend(loc='lower right', prop={'size':TEXTSIZE})
	ax.set_ylim([-1.5,1.5])
	cm = np.real(cm)
	ax.text(.15, .95, r'$m\ddot{x}+c\dot{x}+kx=0$,   $m=10^3$,   $c='+'{:2.1f}'.format(cm)+'$,   $k='+'{:2.1f}'.format(km)+'$', ha='left', va='top', transform=ax.transAxes, size=20)
	
	plt.show()
def suspension_model():
	rolldice = Button(description='ROLL THE DICE', tooltip='generate a random set of parameters for the model')
	
	out = Output()

	def on_button_clicked(b):
		with out:
			clear_output(True)
			plot_suspension_model()

	rolldice.on_click(on_button_clicked)
	
	with out:
		plot_suspension_model()
	
	#rolldice.on_click(lambda x: plot_suspension_model())
	return VBox([out, rolldice])
	
# plot data, car suspension model, and sum of squares	
def plot_suspension_model2(cm,km,weights=False):
	f = plt.figure(figsize=(12,6))
	ax = plt.axes([0.1,0.1,0.8,0.8])
	
	td,xd,pars = get_suspension_data(noisy=True)
	m,x0,dxdt0 = pars
	
	for t in ax.get_xticklabels()+ax.get_yticklabels():
		t.set_fontsize(TEXTSIZE)
		
	ax.set_xlabel('time [s]',size=TEXTSIZE)
	ax.set_ylabel('displacement [m]', size=TEXTSIZE)
	
	t = np.linspace(0,100,201)
	x = suspension(t,cm,km,*pars)
	xm = suspension(td,cm,km,*pars)
	
	ax.fill_between([20,60],[-1.5,-1.5],[1.5,1.5], color=[0.9,0.9,0.9])
	
	for tdi,xdi,xmi in zip(td,xd,xm):
		if not weights:
			ax.plot([tdi,tdi], [xdi,xmi], 'r-', lw = 2)
		else:
			if 20<tdi<60:
				ax.plot([tdi,tdi], [xdi,xmi], 'r-', lw = 1, alpha=0.5)
			else:
				ax.plot([tdi,tdi], [xdi,xmi], 'r-', lw = 2)
	ax.plot([tdi,tdi], [xdi,xmi], 'r-', lw = 2, label = 'misfit')
	
	ax.plot(td,xd,'bo',mfc='w',mew=1.5,label='data')
	ax.plot(t,x,'k-',mfc='w',mew=1.5,label='model')
	
	ax.legend(loc='lower right', prop={'size':TEXTSIZE})
	ax.set_ylim([-1.5,1.5])
	cm = np.real(cm)
	ax.text(.42, .95, r'malfunctioning recording', ha='center', va='top', transform=ax.transAxes, size=TEXTSIZE)
	ax.arrow(20,1.1, 40,0, head_length=1.5, head_width=0.07, color='k', length_includes_head=True)
	ax.arrow(60,1.1,-40,0, head_length=1.5, head_width=0.07, color='k', length_includes_head=True)
	
	# an array of weights
	sigma = np.ones(len(td))
	if weights:
		t1,t2 = [20,60]
		inds = np.where((td>t1)&(td<t2))
		sigma[inds] = 100.
	sigma /= np.sum(sigma**-2)
	
	#
	#S1 = 1.e32
	#S2 = 1.e32
	#sigma2 = np.ones(len(td))
	#sigma2 /= np.sum(sigma2**-2)
	#for c in range(10,110,10):
	#	for k in range(2,29,2):
	#		S = np.sum(((suspension(td,c,k,*pars)-xd)/sigma)**2)
	#		if S<S1:
	#			save1 = [copy(c),copy(k)]
	#			S1 = copy(S)
	#		S = np.sum(((suspension(td,c,k,*pars)-xd)/sigma2)**2)
	#		if S<S2:
	#			save2 = [copy(c),copy(k)]
	#			S2 = copy(S)
	#print(S1,save1)
	#print(S2,save2)
	
	S = np.sum(((xm-xd)/sigma)**2)
	
	ax.text(.70, .95, r'$S('+'{:d},{:d}'.format(cm,km)+')='+'{:3.2e}'.format(S)+'$', ha='left', va='top', transform=ax.transAxes, size=TEXTSIZE, color = 'r')
	plt.show()
def sum_of_squares():
	
	csldr = IntSlider(value = 50, description='$c$', min=10, max = 100, step=10, continuous_update=False)
	ksldr = IntSlider(value = 13, description='$k$', min=2, max = 27, step=2, continuous_update=False)
	wgts = Checkbox(value = False, description='downweight bad measurements')
	return VBox([HBox([csldr, ksldr, wgts]), interactive_output(plot_suspension_model2, {'cm':csldr,'km':ksldr,'weights':wgts})])	
	
# adhoc calibration exercise
def habanero_eqs(model = None):
	f = plt.figure(figsize=(12,6))
	ax = plt.axes([0.1,0.1,0.8,0.8])
	time,distance = np.genfromtxt('eqs.txt', delimiter = ',', skip_header = 1).T
	ax.plot(time, distance, 'kx', ms = 2)    # plot the analytical solution
	
	if model is not None:
		ax.plot(model[0], model[1], 'b-')
	
	ax.set_xlabel('time [days]', size = TEXTSIZE)
	ax.set_ylabel('distance from well [m]', size = TEXTSIZE)
	ax.set_xlim([10,30])
	ax.set_ylim([0,1650])
	ax.set_title('Earthquake locations during well stimulation', size = TEXTSIZE);
	for tick in ax.get_xticklabels()+ax.get_yticklabels(): tick.set_fontsize(TEXTSIZE)
	plt.show()
# tennis ball example
def vi(h0,g,e,i):  
    # velocity after ith bounce
    return np.sqrt(2*g*h0*e**i)
def u(t,ti,vi,g): 
    # position of bouncing ball
    return -g/2*t**2+(g*ti+vi)*t-vi*ti-g/2*ti**2
def ti1(ti, vi, g):  
    # time between bounces
    return ti+2*vi/g
def tennis_ball_model(g, h0, e):
	# parameters
	Nbounces = 10    # number of bounces to compute
	
	# compute for first "half" bounce
	v0 = vi(h0,g,e,0)
	t0 = -v0/g
	tv = np.linspace(0, -t0, 101)   # time vector
	uv = u(tv,t0,v0,g)              # position vector
	ti = ti1(t0,v0,g)               # update bounce time

    # loop through number of bounces
	for i in range(1,Nbounces+1):
		vv = vi(h0,g,e,i)               # velocity at start of bounce
		ti = ti1(ti,vv,g)               # update bounce-time
		tnew = np.linspace(tv[-1], ti, 101)   # time vector for bounce
		unew = u(tnew,tv[-1],vv,g)      # position during bounce
		tv = np.concatenate((tv,tnew))  # update full time vector
		uv = np.concatenate((uv,unew))  # update full position vector
		
	return tv, uv
def tennis_ball_plot(model = None, data=None, threshold = None, tmax = 12, umax=10):
	f = plt.figure(figsize=(12,6))
	ax = plt.axes([0.1,0.1,0.8,0.8])
	
	if data is not None:
		ax.plot(data[0], data[1], 'ro')    # plot the analytical solution
	
	if model is not None:
		tm, um = tennis_ball_model(model[0], model[1], model[2])
		ax.plot(tm, um, 'b-')
		
	if model is not None and data is not None:
		udi = np.interp(data[0],tm,um)
		S = np.sum((udi-data[1])**2)
		ax.text(0.95, 0.95, 'objective function = %2.1f'%S, transform=ax.transAxes, ha = 'right', va = 'top', size = 20)
		
	if threshold is not None:
		ax.axhline(threshold, color='k', linestyle='--')
	
	ax.set_xlabel('time', size = TEXTSIZE)
	ax.set_ylabel('height [m]', size = TEXTSIZE)
	ax.set_xlim([0,tmax]); ax.set_ylim([0,umax])
	for tick in ax.get_xticklabels()+ax.get_yticklabels(): tick.set_fontsize(TEXTSIZE)
	plt.show()
# parameter space figures
def r(X,Y,p): 
	return (p[0]-np.exp(-((X-p[3])**2/p[5]+(Y-p[4])**2/p[6])))*(1-(X/p[1])**p[2])*(1+(Y/p[1])**p[2])
def plot_parameter_space(ic,ik):
	x = np.linspace(0,1,31)
	y = np.linspace(0,1,31)

	xm = np.mean(x)*0.8
	ym = np.mean(y)*1.2
	sx = 0.02*3.
	sy = 0.04*3.

	ymin,ymax = [0.15,0.85]
	i1 = np.argmin(abs(y-ymin))
	i2 = np.argmin(abs(y-ymax))
	y2 = y[i1:i2+1]

	[X,Y] = np.meshgrid(x,y)
	[X2,Y2] = np.meshgrid(x,y2)
	
	fig = plt.figure(figsize=[8,8])
	ax = fig.add_subplot(111, projection='3d')
	ax.plot3D([0,1],[y[i1], y[i1]],[7,7], 'k:', lw = 1, zorder = 1)
	ax.plot3D([0.72,1],[y[i2], y[i2]],[7,7], 'k:', lw = 1, zorder = 1)
	
	n = 2
	a = 4
	obs = 10.
	p = [obs, a, n, xm, ym, sx, sy]
	
	xi = x[ic]
	yi = y[ik]
	zi = r(xi,yi,p)
	
	ax.plot3D([xi,xi,xi,xi,x[-1]],[yi,yi,y[0],yi,yi],[zi,7,7,7,7],'k-', ms = 10, mew=2, zorder=2)
	
	ax.plot3D([xi,],[yi,],[7,],'ks', ms = 8, mfc='w', mew=2, zorder=2)
	ax.text3D(xi,yi,7.3,'\n'+r'$\theta$', va= 'top',ha='center',size=TEXTSIZE)
	
	ax.plot_wireframe(X, Y, r(X,Y,p), lw = 0.5, color = 'k', zorder=4)
	ax.plot_surface(X2, Y2, r(X2,Y2,p), rstride=1, cstride=1,cmap=colmap.Oranges, lw = 0.5, zorder=3)
	
	ax.plot3D([xi,],[yi,],[zi,],'kx', ms = 10, mew=2,zorder=5)
	ax.text3D(xi,yi,zi+0.3,r'$S(\theta)$', va= 'bottom',ha='center',size=TEXTSIZE, backgroundcolor='w', bbox={'pad':0.1,'color':'w'})
			
	ax.set_zlim([7,13])
	ax.set_xlim([0,1])
	ax.set_ylim([0,1])

	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_zticklabels([])

	ax.set_xlabel('$c$', size=TEXTSIZE)
	ax.set_ylabel('$k$', size=TEXTSIZE)
	ax.set_zlabel(r'$S(\mathbf{\theta})$', rotation = 180., size=TEXTSIZE)
	
	plt.show()
	
def parameter_space():	
	csldr = IntSlider(value = 10, description='$c$', min=0, max = 31, step=1, readout=False, continuous_update=False)
	ksldr = IntSlider(value = 10, description='$k$', min=4, max = 25, step=1, readout=False, continuous_update=False)
	return VBox([HBox([csldr, ksldr]), interactive_output(plot_parameter_space, {'ic':csldr,'ik':ksldr})])
	
def plot_parameter_space2(ic,ik,check):
	fig = plt.figure(figsize=[8,8])
	ax = plt.axes([0.1,0.1,0.8,0.8])
	x = np.linspace(0,1,101)
	y = np.linspace(0,1,101)

	xm = np.mean(x)*0.8
	ym = np.mean(y)*1.2
	sx = 0.02*3.
	sy = 0.04*3.

	ymin,ymax = [0.15,0.85]
	i1 = np.argmin(abs(y-ymin))
	i2 = np.argmin(abs(y-ymax))
	y2 = y[i1:i2+1]

	[X,Y] = np.meshgrid(x,y)
	[X2,Y2] = np.meshgrid(x,y2)
	
	n = 2
	a = 4
	obs = 10.
	p = [obs, a, n, xm, ym, sx, sy]

	xi = ic/20*x[-1]+x[0]
	yi = ik/20*y[-1]+y[0]
	zi = r(xi,yi,p)
	if check:
		xi1 = (ic+1)/20*x[-1]+x[0]
		dxi1 = 1/20*x[-1]
		yi1 = (ik+1)/20*y[-1]+y[0]
		dyi1 = 1/20*y[-1]*2.
		zi01 = r(xi1,yi,p)
		zi10 = r(xi,yi1,p)
		s = np.array([(zi01-zi)/dxi1, (zi10-zi)/dyi1])
		s = -s/np.sqrt(np.dot(s,s))
	
	
	ax.set_xticks([xi,])
	ax.set_yticks([yi,])
	ax.set_yticklabels(['{:2.1f}'.format(yi*2.)])
	ax.set_xlabel('c', size=TEXTSIZE)
	ax.set_ylabel('k', size=TEXTSIZE)
	if check:
		ax.arrow(xi,yi,s[0]/20., s[1]/20., head_length = 0.015, head_width=0.015, color = 'k')
		ax.plot([x[0], xi, xi],[yi,yi,y[0]],'k:', lw=0.5, zorder=3)
	else:
		ax.plot([x[0], xi, xi],[yi,yi,y[0]],'k-',zorder=3)
	ax.plot(xi,yi,'kx',zorder=3, ms=10, mew=2)
	
	for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontsize(TEXTSIZE)
		
	levels = np.linspace(np.min(r(X,Y,p)), np.max(r(X,Y,p)), 11)
	ax.contourf(X2,Y2,r(X2,Y2,p), cmap = colmap.Oranges, levels = levels)
	ax.contour(X,Y,r(X,Y,p), levels = levels, colors = 'k', linewidths = 1)
	
	if check:
		txt = r'-$\hat{\mathbf{s}}$='+'[{:0.2f},{:0.2f}]'.format(s[0],s[1])
	else:
		txt = '$S$={:4.3f}'.format(zi)
	ax.text(0.05,0.95,txt, ha='left', va='top', transform=ax.transAxes, size=20, backgroundcolor='w')
	plt.show()
	
def sensitivity():
	csldr = IntSlider(value = 10, description='$c$', min=0, max = 20, step=1, readout=False, continuous_update=False)
	ksldr = IntSlider(value = 10, description='$k$', min=3, max = 17, step=1, readout=False, continuous_update=False)
	check = Checkbox(value = False, description='check my answer')
	return VBox([HBox([csldr, ksldr, check]), interactive_output(plot_parameter_space2, {'ic':csldr,'ik':ksldr, 'check':check})])

def plot_parameter_space3(ic,ik,N=0,alpha=0.03):
	fig = plt.figure(figsize=[8,8])
	ax = plt.axes([0.1,0.1,0.8,0.8])
	x = np.linspace(0,1,101)
	y = np.linspace(0,1,101)

	xm = np.mean(x)*0.8
	ym = np.mean(y)*1.2
	sx = 0.02*3.
	sy = 0.04*3.

	ymin,ymax = [0.15,0.85]
	i1 = np.argmin(abs(y-ymin))
	i2 = np.argmin(abs(y-ymax))
	y2 = y[i1:i2+1]

	[X,Y] = np.meshgrid(x,y)
	[X2,Y2] = np.meshgrid(x,y2)
	
	n = 2
	a = 4
	obs = 10.
	p = [obs, a, n, xm, ym, sx, sy]

	xi = ic/20*x[-1]+x[0]
	yi = ik/20*y[-1]+y[0]
	ax.plot([x[0], xi, xi],[yi,yi,y[0]],'k:', lw=0.5, zorder=3)
	ax.set_xticks([xi,])
	ax.set_yticks([yi,])
	ax.set_yticklabels(['{:2.1f}'.format(yi*2.)])
	
	x = [xi,]
	y = [yi,]
	ax.plot(xi,yi,'kx',lw=2, ms=10, mew=2)
	
	for i in range(N):
		dx = 0.01
		zi = r(xi,yi,p)
		s = np.array([(r(xi+dx,yi,p)-zi)/dx,(r(xi,yi+dx,p)-zi)/dx])
		s = -s/np.sqrt(np.dot(s,s))
		
		xi1,yi1 = np.array([xi,yi]) + alpha*s
		
		ax.arrow(xi,yi,xi1-xi,yi1-yi,color='k',head_length = 0.015, head_width=0.015, length_includes_head=True)
		
		xi = 1.*xi1
		yi = 1.*yi1
	
	for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontsize(TEXTSIZE)
		
	ax.set_xlabel('c', size=TEXTSIZE)
	ax.set_ylabel('k', size=TEXTSIZE)
	
	levels = np.linspace(np.min(r(X,Y,p)), np.max(r(X,Y,p)), 11)
	ax.contourf(X2,Y2,r(X2,Y2,p), cmap = colmap.Oranges, levels = levels)
	ax.contour(X,Y,r(X,Y,p), levels = levels, colors = 'k', linewidths = 1)
	
	plt.show()
		
def gradient_descent():
	csldr = IntSlider(value = 10, description='$c$', min=0, max = 20, step=1, readout=False, continuous_update=False)
	ksldr = IntSlider(value = 10, description='$k$', min=3, max = 17, step=1, readout=False, continuous_update=False)
	Nsldr = IntSlider(value = 0, description='steps', min=0, max = 10, step=1, continuous_update=False)
	asldr = FloatSlider(value = 0.05, description=r'$\alpha$', min=0, max = 0.10, step=0.02, continuous_update=False)
	return VBox([HBox([csldr, ksldr, Nsldr, asldr]), interactive_output(plot_parameter_space3, {'ic':csldr,'ik':ksldr, 'N':Nsldr, 'alpha':asldr})])
	
	
	
	
	
	
	
	
	