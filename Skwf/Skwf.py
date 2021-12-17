import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.integrate as integr
import scipy.constants as cnt
from matplotlib.patches import Circle
import imageio
import scipy.constants as const
from threading import Thread
import os
from scipy.optimize import curve_fit
from collections import deque
from joblib import Parallel, delayed
import time
from numba import jit
import random
import pyroot
import scipy.special as spec

#x = np.linspace(-np.pi,np.pi,111)
#y1 = np.cos(x)
#y2 = np.sin(x)
#plt.plot(x, y1, color="r", linestyle="-", linewidth=2)
#plt.plot(x, y2, "g--", lw=1)
#
#plt.plot(x, y2, "ro", lw=2)
#plt.plot(x, x, "r--", x, x**2, "bs",\
#x, x**3, "g^")
#
#plt.plot(x, x, x, x**2, x, x**3,\
#color="r", linestyle= "--", linewidth=2.0)
#
#plt.plot(x, x, x, x**2, x, x**3)
#
#plt.xlabel("x dana", fontsize=20)
#plt.ylabel("y wynik", fontsize=20)
#plt.title("Funkcje: sinus i cosinus", fontsize=24)
#plt.grid(True)
#plt.legend( ("cos(x)","sin(x)") )
#plt.savefig("xy.png", format="png")
#plt.show()

#Ćw 2

#def Brown_1D():
#	cmap = plt.get_cmap('Set1')
#	N = 10
#	rnd_arr = np.random.randn(N,100)
#	rnd_arr[:,0] = 0
#	summed = np.cumsum(rnd_arr, axis=1)
#	opis = []
#	for i in range(1):
#		plt.plot(summed[i,:], color=cmap(i))
#		opis.append("Cząstka: {}".format(i))
#	plt.legend(opis)
#	plt.show()
#	for i in range(N):
#		plt.plot(summed[i,:], color=cmap(i))
#		opis.append("Cząstka: {}".format(i))
#	plt.legend(opis)
#	plt.show()
#	return
#
#def Brown_2D():
#	cmap = plt.get_cmap('Set1')
#	rnd_arr = np.random.randn(2,100)
#	rnd_arr[:,0] = 0
#	summed = np.cumsum(rnd_arr, axis=1)
#	opis = []
#	plt.plot(summed[0,:],summed[1,:], color=cmap(0))
#	opis.append("Cząstka")
#	plt.legend(opis)
#	plt.show()
#	return
#
#def hist_final():
#	for T in [10,50,100]:
#		hist(T)
#
#def hist(T):
#	N = 100000
#	rnd_arr = np.random.randn(N,T)
#	final_positions = np.sum(rnd_arr, axis=1)
#	plt.hist(final_positions, bins=50, density=True, facecolor='g', alpha=0.75)
#	m = 40#np.max(np.abs(final_positions))
#	pos = np.linspace(-m,m,200)
#	plt.plot(pos, np.exp(-pos ** 2 / (2 * T)) / np.sqrt(2 * np.pi * T))
#	plt.show()
#
#def std_dev():
#	for i in range(1,6):
#		T = 100
#		N = 10 ** i
#		rnd_arr = np.random.randn(N,T + 1)
#		rnd_arr[:,0] = 0
#		tp = range(T + 1)
#		summed = np.cumsum(rnd_arr, axis=1)
#		stddev = np.var(summed, axis=0)
#		plt.plot(tp,tp,label="Analityczny")
#		plt.plot(tp,stddev,label="Symulacyjny")
#		plt.legend()
#		plt.show()
#
#Ćw 3
def Euler():
	G = 1e-2
	M = 500
	m = 0.1
	dt = 1e-3
	T_k = 50
	N = int(T_k / dt)
	p = [np.array([0.,0.1])]
	x = [np.array([2.,0.])]
	for i in range(1,N + 1):
		r = np.sqrt(np.sum(x[-1] ** 2))
		F = -G * M * m / r ** 3 * x[-1]
		p_l = p[-1] + F * dt
		p.append(p_l)
		x_l = x[-1] + p[-1] / m * dt + 0.5 * F / m * dt ** 2
		x.append(x_l)
	t = np.linspace(0.,T_k,N + 1)
	x_n = np.stack(x, axis=0)
	v_n = np.stack(p, axis=0) / (m)
	V = -G * M * m / np.hypot(x_n[:,0],x_n[:,1])
	E_k = 0.5 * m * (v_n[:,0] ** 2 + v_n[:,1] ** 2)
	plt.plot(x_n[:,0],x_n[:,1])
	plt.show()
	plt.plot(t,V)
	plt.plot(t,E_k)
	plt.plot(t,E_k + V)
	plt.legend(["V","E_k","E_c"])
	plt.show()

def Verlet():
	G = 1e-2
	M = 500
	m = 0.1
	dt = 1e-3
	T_k = 50
	N = int(T_k / dt)
	p = [np.array([0.,0.1])]
	x_0 = np.array([2.,0.])
	x__1 = x_0 - dt * p[-1] / m
	x = [x__1,x_0]
	for i in range(1,N + 1):
		r = np.sqrt(np.sum(x[-1] ** 2))
		F = -G * M * m / r ** 3 * x[-1]
		x_l = 2 * x[-1] - x[-2] + F / m * dt ** 2
		x.append(x_l)
	x_n = np.stack(x, axis=0)
	v_n = (x_n[2:,:] - x_n[:-2,:]) / (2 * dt)
	t = np.linspace(dt,T_k - dt,N)
	V = -G * M * m / np.hypot(x_n[:,0],x_n[:,1])
	E_k = 0.5 * m * (v_n[:,0] ** 2 + v_n[:,1] ** 2)
	plt.plot(x_n[:,0],x_n[:,1])
	plt.show()
	plt.plot(t,V[1:-1])
	plt.plot(t,E_k)
	plt.plot(t,E_k + V[1:-1])
	plt.legend(["V","E_k","E_c"])
	plt.show()

def Zabka():
	G = 1e-2
	M = 500
	m = 0.1
	dt = 1e-3
	T_k = 50
	N = int(T_k / dt)
	x = [np.array([2.,0.])]
	r = np.sqrt(np.sum(x[-1] ** 2))
	v = [np.array([0.,0.1]) / m - 0.5 * (-G * M / r ** 3 * x[-1]) * dt]
	for i in range(1,N + 1):
		r = np.sqrt(np.sum(x[-1] ** 2))
		F = -G * M / r ** 3 * x[-1]
		v_l = v[-1] + F * dt
		x_l = x[-1] + v_l * dt
		x.append(x_l)
		v.append(v_l)
	t = np.linspace(0.,T_k,N + 1)
	x_n = np.stack(x, axis=0)
	v_n = np.stack(v, axis=0)
	V = -G * M * m / np.hypot(x_n[:,0],x_n[:,1])
	E_k = 0.5 * m * (v_n[:,0] ** 2 + v_n[:,1] ** 2)
	plt.plot(x_n[:,0],x_n[:,1])
	plt.show()
	plt.plot(t,V)
	plt.plot(t,E_k)
	plt.plot(t,E_k + V)
	plt.legend(["V","E_k","E_c"])
	plt.show()

def osem():
	G = 1
	m = 1
	dt = 1e-3
	T_k = 40
	N = int(T_k / dt)
	x = [[[],[]],[[],[]],[[],[]]]
	v = [np.array([]),np.array([]),np.array([])]
	v[2] = np.array([0.93240737,0.86473146])
	v[0] = -0.5 * v[2]
	v[1] = -0.5 * v[2]
	r = [np.array([]),np.array([]),np.array([])]
	r[0] = np.array([0.97000436,-0.24308753])
	r[1] = -r[0]
	r[2] = np.array([0.,0.])
	dv = [np.array([]),np.array([]),np.array([])]
	f = [[],[],[]]
	for j in range(1,N + 1):
		r_12 = np.sqrt(np.sum((r[0] - r[1]) ** 2))
		r_13 = np.sqrt(np.sum((r[0] - r[2]) ** 2))
		r_23 = np.sqrt(np.sum((r[1] - r[2]) ** 2))
		f[0].append(r_12)
		f[1].append(r_13)
		f[2].append(r_23)
		F_12 = G * m * (r[1] - r[0]) / (r_12 ** 3)
		F_13 = G * m * (r[2] - r[0]) / (r_13 ** 3)
		F_23 = G * m * (r[2] - r[1]) / (r_23 ** 3)
		dv[0] = (F_12 + F_13) * dt
		dv[1] = (-F_12 + F_23) * dt
		dv[2] = (-F_13 - F_23) * dt
		for i in range(3):
			r[i] += (v[i] * dt + 0.5 * dv[i] * dt)
			v[i] += dv[i]
			x[i][0].append(r[i][0])
			x[i][1].append(r[i][1])
	for i in range(3):
		plt.plot(x[i][0],x[i][1])
	plt.show()
	for i in range(3):
		plt.plot(f[i])
	plt.show()

#Ćw 4
def f(d):
	return 48 * ((1 / d) ** 13 - 0.5 * (1 / d) ** 7)

def u(d):
	return 4 * ((1 / d) ** 12 - (1 / d) ** 6)

def moving_average(a, n=25) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def particles(temp):
	particleRow = 16
	particleNumber = particleRow ** 2
	boxsize = 2. * particleRow
	V = boxsize ** 2
	eps = 1.0
	sigma = 1.0
	promien = 0.5
	dt = 0.005
	#temp = 2.5
	kB = 1
	dx = boxsize / particleRow
	delta = 1e-20
	T = 100
	r_c = 2.5
	N_t = int(T / dt)
	plot_step = N_t / int(T / 1e-1)
	plotting = False
	lin = np.tile(np.linspace(0,particleRow - 1, particleRow), (particleRow,1))
	pos = np.transpose(np.reshape(np.stack([lin, np.transpose(lin)]), (2,particleNumber))) * dx + dx / 2
	v = np.random.random((particleNumber,2)) - 0.5
	sumv2 = np.sum(v ** 2) / (2 * particleNumber)
	fs = np.sqrt(temp / sumv2)
	v*=fs
	pos_tiled = np.tile(pos, (particleNumber, 1, 1))
	T_arr = []
	p_arr = []
	E_arr = []
	E_kin_arr = []
	E_pot_arr = []
	cycles = N_t
	T_ext = 2 * temp
	try:
		os.mkdir(str(temp))
	except Exception:
		pass
	for tp in range(N_t):
		pos_tiled = np.tile(pos, (particleNumber, 1, 1))
		temp_r_ij = pos_tiled - np.transpose(pos_tiled, axes=(1,0,2))
		t_abs = np.abs(temp_r_ij)
		t_sgn = np.sign(temp_r_ij)
		diff = - t_sgn * boxsize + temp_r_ij
		r_ij = np.where(t_abs < boxsize / 2, temp_r_ij, diff)
		d = np.linalg.norm(r_ij, axis = 2) + delta
		norm_r_ij = np.stack([r_ij[:,:,0] / d, r_ij[:,:,1] / d])
		F_mag = f(d)#np.where(d <= r_c, f(d),0.)
		F = norm_r_ij * F_mag
		F_s = np.transpose(np.sum(F, axis=1))
		F_scaled = F_s * dt
		v_virt = v - F_scaled / 2
		kin = np.sum(v_virt ** 2) / 2
		d_copy = d
		np.fill_diagonal(d_copy,2 * r_c)
		pot = np.sum(np.where(d <= r_c, u(d_copy) - u(r_c),0.))
		E_arr.append(kin + pot)
		E_kin_arr.append(kin)
		E_pot_arr.append(pot)
		T = kin / (particleNumber * kB)
		eta = np.sqrt(T_ext / T)
		#eta = np.sqrt(T_ext / T)
		v = (2 * eta - 1) * v + eta * F_scaled
		pos += v * dt
		pos %= boxsize
		np.fill_diagonal(F_mag,0.)
		P = particleNumber * kB * T / V + np.sum(F_mag * d) / (2 * V)
		T_arr.append(T)
		p_arr.append(P)
		if (tp % plot_step == 0 and plotting):
			plt.clf() # wyczyść obrazek
			F = plt.gcf() # zdefiniuj nowy
			nStr = str(tp)
			nStr = nStr.rjust(5,'0')
			for r in pos: # pętla po cząstkach
				a = plt.gca() # ‘get current axes' (to add smth to them)
				cir = Circle(((r[0] + boxsize / 2.) % boxsize, (r[1] + boxsize / 2.) % boxsize), radius=promien) # kółko tam gdzie jest cząstka
				a.add_patch(cir) # dodaj to kółko do rysunku
				plt.plot() # narysuj
			plt.xlim((0,boxsize)) # obszar do narysowania
			plt.ylim((0,boxsize))
			F.set_size_inches((6,6)) # rozmiar rysunku
			plt.title("Symulacja gazu Lennarda-Jonesa, krok " + nStr)
			plt.savefig(str(temp) + '/img' + nStr + '.png')
	images = []
	if plotting:
		for tp in range(N_t):
			if (tp % plot_step == 0):
				nStr = str(tp) #nagraj na dysk – numer pliku z 5 cyframi, na początku zera, np 00324.png
				nStr = nStr.rjust(5,'0')
				images.append(imageio.imread(str(temp) + '/img' + nStr + '.png'))
		imageio.mimsave(str(temp) + '/movie.gif', images)#, duration=10)
	plt.clf()
	E_min = np.min(E_arr)
	E_s = E_min + (E_arr[0] - E_min) * 0.025
	#print(E_ma.shape)
	print(np.argmax(E_arr < E_s))
	#plt.plot(np.array(T_arr), label="T")
	##plt.plot((np.array(p_arr[1:]) + np.array(p_arr[:-1])) / 2, label="p")
	#plt.plot(T_arr, label="T")
	#plt.plot(p_arr, label="p")
	plt.plot(E_arr, label="E")
	plt.plot(E_kin_arr, label="E_kin")
	plt.plot(E_pot_arr, label="E_pot")
	plt.legend()
	plt.show()
	return

def parts_thr():
	thrs = []
	for x in [0.1, 0.7, 2.5]:
		particles(x)
		break

#Ćw 5
def duff(t,data, a,b,c,omega,f):
	x,v,d = data[0],data[1],data[2]
	return [v, b * x - a * x ** 3 - c * v + f * np.cos(d), omega]

def Lorenz(t,data,  sigma, b, r):
	x,y,z = data[0],data[1],data[2]
	return [sigma * (y - x),(r - z) * x - y,x * y - b * z]

def zad0():
	t_end = 400.
	ts = np.arange(0.,t_end,t_end * 10 ** (-6))
	res = integr.solve_ivp(Lorenz, t_span=[0.,t_end], y0=[0,0.5,1], t_eval=ts, args=(10.,8. / 3,28))
	plt.plot(res["y"][0,:],res["y"][1,:], "g-", label="Trajektoria")
	plt.legend()
	plt.show()

def zad1(pltos=False,f=0.2,z2=False):
	t_end = 400.
	ts = np.arange(0.,t_end,t_end * 10 ** (-4))
	res = integr.solve_ivp(duff, t_span=[0.,t_end], y0=[0.5,0.1,0], t_eval=ts, args=(1.,1.,0.2,0.213 * 2 * cnt.pi,f))
	if pltos:
		plt.plot(ts,res["y"][0,:], "g-", label="położenie")
		plt.title("f=" + str(f))
		plt.xlabel("t")
		plt.xlabel("x")
		plt.legend()
		plt.show()
		plt.plot(ts,res["y"][1,:], "g-", label="prędkość")
		plt.title("f=" + str(f))
		plt.xlabel("t")
		plt.xlabel("v")
		plt.legend()
		plt.show()
	if z2:
		mx = np.argmax(ts > 200)
		plt.plot(res["y"][0,mx:],res["y"][1,mx:], "g-", label="Trajektoria w p. faz.")
		plt.title("f=" + str(f))
		plt.xlabel("x")
		plt.xlabel("v")
		plt.legend()
		plt.show()
	else:
		plt.plot(res["y"][0,:],res["y"][1,:], "g-", label="Trajektoria w p. faz.")
		plt.title("f=" + str(f))
		plt.xlabel("x")
		plt.xlabel("v")
		plt.legend()
		plt.show()

def zad2():
	#for f in range(3,10):
	#	zad1(False,0.1*2**f,True)
	for f in [0.1,0.3,0.32650305, 0.33]:
		zad1(False,f,True)
	return

def zad3():
	omega = 0.213 * 2 * cnt.pi
	n = 20000
	t_end = n / 0.213
	ts = np.arange(0.,t_end, 1 / 0.213)
	print(ts.shape)
	res = integr.solve_ivp(duff, t_span=[0.,t_end], y0=[0,0.15,0], t_eval=ts, args=(1.,1.,0.07,omega,0.3))
	plt.scatter(res["y"][0,:],res["y"][1,:], s=3, c="b", lw=0, marker="o", label="punkty")
	plt.xlabel("x")
	plt.xlabel("v")
	plt.legend()
	plt.show()
	return

def frakt_fit(r,D_f,c):
	return D_f * np.log(2) * r + c

def frakt_sierp():
	m = [[],[],[]]
	prob = [1. / 3,1. / 3,1. / 3]
	N = int(1e6)
	x_rec = np.zeros(N + 1)
	y_rec = np.zeros(N + 1)
	m[0] = [0.5, 0, 0, 0.5, 0.25, np.sqrt(3.) / 4]
	m[1] = [0.5, 0, 0, 0.5, 0.0, 0]
	m[2] = [0.5, 0, 0, 0.5, 0.5, 0]
	choices = np.random.choice([0,1,2], size=N, p=prob)
	for i in range(N):
		x_rec[i + 1] = m[choices[i]][0] * x_rec[i] + m[choices[i]][1] * y_rec[i] + m[choices[i]][4]
		y_rec[i + 1] = m[choices[i]][2] * x_rec[i] + m[choices[i]][3] * y_rec[i] + m[choices[i]][5]
	plt.scatter(x_rec,y_rec, s=1, c="b", lw=0, marker="o", label="punkty")
	plt.xlabel("x")
	plt.xlabel("y")
	plt.legend()
	plt.show()
	nrs = []
	bs = []
	for r in range(1,14):
		H,_,_ = np.histogram2d(x_rec[:int(N / 4)],y_rec[:int(N / 4)],bins=2 ** r)
		nr = np.sum(H > 0)
		nrs.append(nr)
		bs.append(r)
	plt.plot(bs,nrs,".r")
	plt.yscale("log")
	plt.show()
	nrs = []
	bs = []
	for r in range(1,14):
		H,_,_ = np.histogram2d(x_rec,y_rec,bins=2 ** r)
		nr = np.sum(H > 0)
		nrs.append(nr)
		bs.append(r)
	par,cov = curve_fit(frakt_fit,bs[:-3],np.log(nrs[:-3]),p0=(1,1))
	print(par[0], np.sqrt(cov[0,0]))
	print(par[1], np.sqrt(cov[1,1]))
	plt.plot(bs,nrs,".r")
	plt.plot(bs,np.exp(frakt_fit(np.array(bs),*par)),"-g")
	plt.yscale("log")
	plt.show()
	return

def frakt_barn():
	m = [[],[],[],[]]
	prob = [0.03,0.11,0.13,0.73]
	N = int(1e6)
	x_rec = np.zeros(N + 1)
	y_rec = np.zeros(N + 1)
	m[0] = [0.001, 0.0, 0.0, 0.16, 0.0, 0.0]
	m[1] = [-0.15, 0.28, 0.26, 0.24, 0.0, 0.44]
	m[2] = [0.2,-0.26, 0.23, 0.22, 0.0, 1.6]
	m[3] = [0.85, 0.04,-0.04, 0.85, 0.0, 1.6]
	choices = np.random.choice([0,1,2,3], size=N, p=prob)
	for i in range(N):
		x_rec[i + 1] = m[choices[i]][0] * x_rec[i] + m[choices[i]][1] * y_rec[i] + m[choices[i]][4]
		y_rec[i + 1] = m[choices[i]][2] * x_rec[i] + m[choices[i]][3] * y_rec[i] + m[choices[i]][5]
	plt.scatter(x_rec,y_rec, s=1, c="b", lw=0, marker="o", label="punkty")
	plt.xlabel("x")
	plt.xlabel("y")
	plt.legend()
	plt.show()
	nrs = []
	bs = []
	for r in range(1,14):
		H,_,_ = np.histogram2d(x_rec[:int(N / 4)],y_rec[:int(N / 4)],bins=2 ** r)
		nr = np.sum(H > 0)
		nrs.append(nr)
		bs.append(r)
	plt.plot(bs,nrs,".r")
	plt.yscale("log")
	plt.show()
	nrs = []
	bs = []
	for r in range(1,14):
		H,_,_ = np.histogram2d(x_rec,y_rec,bins=2 ** r)
		nr = np.sum(H > 0)
		nrs.append(nr)
		bs.append(r)
	par,cov = curve_fit(frakt_fit,bs[:-3],np.log(nrs[:-3]),p0=(1,1))
	print(par[0], np.sqrt(cov[0,0]))
	print(par[1], np.sqrt(cov[1,1]))
	plt.plot(bs,nrs,".r")
	plt.plot(bs,np.exp(frakt_fit(np.array(bs),*par)),"-g")
	plt.yscale("log")
	plt.show()
	return

def frakt_drag():
	m = [[],[]]
	prob = [0.787473,0.212527]
	N = int(1e6)
	x_rec = np.zeros(N + 1)
	y_rec = np.zeros(N + 1)
	m[0] = [0.824074, 0.281482, -0.212346, 0.864198, -1.882290, -0.110607]
	m[1] = [0.088272, 0.520988, -0.463889, -0.377778, 0.785360, 8.095795]
	choices = np.random.choice([0,1], size=N, p=prob)
	for i in range(N):
		x_rec[i + 1] = m[choices[i]][0] * x_rec[i] + m[choices[i]][1] * y_rec[i] + m[choices[i]][4]
		y_rec[i + 1] = m[choices[i]][2] * x_rec[i] + m[choices[i]][3] * y_rec[i] + m[choices[i]][5]
	plt.scatter(x_rec,y_rec, s=1, c="b", lw=0, marker="o", label="punkty")
	plt.xlabel("x")
	plt.xlabel("y")
	plt.legend()
	plt.show()
	nrs = []
	bs = []
	for r in range(1,14):
		H,_,_ = np.histogram2d(x_rec[:int(N / 4)],y_rec[:int(N / 4)],bins=2 ** r)
		nr = np.sum(H > 0)
		nrs.append(nr)
		bs.append(r)
	plt.plot(bs,nrs,".r")
	plt.yscale("log")
	plt.show()
	nrs = []
	bs = []
	for r in range(1,14):
		H,_,_ = np.histogram2d(x_rec,y_rec,bins=2 ** r)
		nr = np.sum(H > 0)
		nrs.append(nr)
		bs.append(r)
	par,cov = curve_fit(frakt_fit,bs[:-3],np.log(nrs[:-3]),p0=(1,1))
	print(par[0], np.sqrt(cov[0,0]))
	print(par[1], np.sqrt(cov[1,1]))
	plt.plot(bs,nrs,".r")
	plt.plot(bs,np.exp(frakt_fit(np.array(bs),*par)),"-g")
	plt.yscale("log")
	plt.show()
	return

def frakt_levy():
	m = [[],[]]
	prob = [0.5,0.5]
	N = int(1e6)
	x_rec = np.zeros(N + 1)
	y_rec = np.zeros(N + 1)
	m[0] = [0.5, -0.5, 0.5, 0.5, 0.0, 0.0]
	m[1] = [0.5, 0.5, -0.5, 0.5, 0.5, 0.5]
	choices = np.random.choice([0,1], size=N, p=prob)
	for i in range(N):
		x_rec[i + 1] = m[choices[i]][0] * x_rec[i] + m[choices[i]][1] * y_rec[i] + m[choices[i]][4]
		y_rec[i + 1] = m[choices[i]][2] * x_rec[i] + m[choices[i]][3] * y_rec[i] + m[choices[i]][5]
	plt.scatter(x_rec,y_rec, s=1, c="b", lw=0, marker="o", label="punkty")
	plt.xlabel("x")
	plt.xlabel("y")
	plt.legend()
	plt.show()
	nrs = []
	bs = []
	for r in range(1,14):
		H,_,_ = np.histogram2d(x_rec[:int(N / 4)],y_rec[:int(N / 4)],bins=2 ** r)
		nr = np.sum(H > 0)
		nrs.append(nr)
		bs.append(r)
	plt.plot(bs,nrs,".r")
	plt.yscale("log")
	plt.show()
	nrs = []
	bs = []
	for r in range(1,14):
		H,_,_ = np.histogram2d(x_rec,y_rec,bins=2 ** r)
		nr = np.sum(H > 0)
		nrs.append(nr)
		bs.append(r)
	par,cov = curve_fit(frakt_fit,bs[:-3],np.log(nrs[:-3]),p0=(1,1))
	print(par[0], np.sqrt(cov[0,0]))
	print(par[1], np.sqrt(cov[1,1]))
	plt.plot(bs,nrs,".r")
	plt.plot(bs,np.exp(frakt_fit(np.array(bs),*par)),"-g")
	plt.yscale("log")
	plt.show()
	return

def perk(L=20, p=0.45, save=False):
	lattice = np.ones((L,L)) * (-1)
	tossing = np.random.random((L,L)) < p
	find_0 = (L // 2,L // 2)
	decs = [np.array([0,1]),np.array([1,0]),np.array([0,-1]),np.array([-1,0])]
	for i in range(L // 2):
		t_sub = tossing[find_0[0] - i:find_0[0] + i + 1,find_0[1] - i:find_0[1] + i + 1]
		l = len(t_sub)
		temp = np.argmax(t_sub)
		temp2 = (temp // l + find_0[0] - i,temp % l + find_0[1] - i)
		if (temp2):
			find_0 = temp2
			break
	coords = np.array(find_0)
	cluster = deque()
	cluster.append(coords)
	lattice[tuple(coords)] = 1
	count = 0
	try:
		os.mkdir('grow')
	except Exception:
		pass
	while(len(cluster) > 0):
		cur_pos = cluster.pop()
		for dec in decs:
			temp_pos = cur_pos + dec
			if (np.all(temp_pos < L) and np.all(temp_pos >= 0) and lattice[tuple(temp_pos)] == -1):
				lattice[tuple(temp_pos)] = tossing[tuple(temp_pos)]
				if lattice[tuple(temp_pos)]:
					cluster.append(temp_pos)
		if save and count % 10 == 0:
			plt.imshow(lattice, interpolation='nearest',cmap='magma')
			plt.grid()
			plt.savefig('grow/img' + str(count) + '.png')
		count += 1
	# also cmap='viridis','inferno','plasma'
	plt.imshow(lattice, interpolation='nearest',cmap='magma')
	plt.grid()
	plt.show()
	return

def perk_prob_elem(L, p):
	lattice = np.ones((L,L)) * (-1)
	tossing = np.random.random((L,L)) < p
	find_0 = (L // 2,L // 2)
	decs = [np.array([0,1]),np.array([1,0]),np.array([0,-1]),np.array([-1,0])]
	for i in range(L // 2):
		t_sub = tossing[find_0[0] - i:find_0[0] + i + 1,find_0[1] - i:find_0[1] + i + 1]
		l = len(t_sub)
		temp = np.argmax(t_sub)
		temp2 = (temp // l + find_0[0] - i,temp % l + find_0[1] - i)
		if (temp2):
			find_0 = temp2
			break
	coords = np.array(find_0)
	cluster = deque()
	cluster.append(coords)
	lattice[tuple(coords)] = 1
	reached = False
	count = 0
	while(len(cluster) > 0):
		cur_pos = cluster.pop()
		count +=1
		for dec in decs:
			temp_pos = cur_pos + dec
			bar = np.all(temp_pos < L) and np.all(temp_pos >= 0)
			reached = reached or not bar
			if reached:
				return (reached, 0)
			if (bar and lattice[tuple(temp_pos)] == -1):
				lattice[tuple(temp_pos)] = tossing[tuple(temp_pos)]
				if lattice[tuple(temp_pos)]:
					cluster.append(temp_pos)
	return (reached, count)

def perk_prob_p(L,p,it):
	count = 0
	avg = 0
	for i in range(it):
		r,c = perk_prob_elem(L,p)
		if r:
			count+=1
		else:
			avg = (avg * (i - count) + c) / float(i - count + 1) 
	return (float(count) / it, avg)

def perk_prob(plot=True):
	Ls = [500]
	avgs = []
	probs_l = []
	prob_c = 50
	col = ['g','r','b']
	probs = np.linspace(0.5,0.7,prob_c)
	for L in Ls:
		results = list(map(list, zip(*Parallel(n_jobs=prob_c)(delayed(perk_prob_p)(L,p,100) for p in probs))))
		avgs.append(results[1])
		probs_l.append(results[0])
	if plot:
		for i in range(len(Ls)):
			plt.plot(probs,probs_l[i],'.' + col[i],label='L = ' + str(Ls[i]))
		plt.legend()
		plt.show()
		for i in range(len(Ls)):
			plt.plot(probs,avgs[i],'.' + col[i],label='L = ' + str(Ls[i]))
		plt.legend()
		plt.show()

def perk_fit(p,p_c,a):
	return 0.5 * (np.tanh((p - p_c) / a) + 1)

def perk_prob_fit(plot=True):
	Ls = np.arange(50,501,50)
	avgs = []
	probs_l = []
	prob_c = 50
	col = ['g','r','b']
	probs = np.linspace(0.55,0.625,prob_c)
	probs_plot = np.linspace(0.55,0.625,100 * prob_c)
	pc_l = []
	a_l = []
	for L in Ls:
		print(L)
		results = list(map(list, zip(*Parallel(n_jobs=prob_c)(delayed(perk_prob_p)(L,p,100) for p in probs))))
		avgs.append(results[1])
		probs_l.append(results[0])
		par,cov = curve_fit(perk_fit, probs, results[0])
		pc_l.append(par[0])
		a_l.append(par[1])
		if (L == 250):
			plt.plot(probs,results[0],".r")
			plt.plot(probs_plot,perk_fit(probs_plot, *par),"-b")
			plt.show()
	print(pc_l)
	print(a_l)
	par = np.polyfit(1 / Ls,pc_l,1)
	plt.plot(1 / Ls,pc_l,".r")
	plt.plot(1 / Ls,np.polyval(par,1 / Ls),"-g")
	plt.title(str(par[1]))
	plt.show()
	plt.xscale("log")
	plt.yscale("log")
	plt.plot(Ls,a_l,".r")
	Ls_l = np.arange(Ls[0],Ls[-1],1)
	plt.plot(Ls_l,Ls_l ** (-1 / (4. / 3)))#a_l[0] / (Ls[0] ** (-1 / (4 / 3))) *
	plt.show()

def DLA(p=0.125):
	try:
		os.mkdir('DLA')
	except Exception:
		pass
	L = 300
	steps = 1000
	particles = 10000
	lattice = np.zeros((L,L))
	moves = [np.array([0,1]),np.array([1,0]),np.array([0,-1]),np.array([-1,0])]
	lattice[L // 2,L // 2] = 1
	angles = np.random.uniform(0.0,2 * np.pi,(particles))
	R = 11.
	counter = 0
	for angle in angles:
		pos = np.array([int(L / 2 + R * np.cos(angle)),int(L / 2 + R * np.sin(angle))])
		choices = np.random.choice([0,1,2,3], size=steps)
		probs = np.random.uniform(0.,1.,(steps)) > p
		for i in range(len(choices)):
			if (any([(lattice[tuple(pos + nb)] == 1) for nb in moves]) and probs[i]):
				lattice[tuple(pos)] = 1
				dist = np.linalg.norm(pos - np.array([L // 2,L // 2])) + 10
				if (dist > R):
					R = dist
				counter += 1
				break
			while (i < len(choices) and lattice[tuple(pos + moves[choices[i]])]):
				i += 1
			if (i == len(choices)):
				break
			pos += moves[choices[i]]
			if np.linalg.norm(pos - np.array([L // 2,L // 2])) > R + 50:
				break
		if counter % 50 == 0:
			plt.imshow(lattice, interpolation='nearest',cmap='magma')
			plt.grid()
			plt.savefig('DLA/img' + str(counter) + '.png')
	plt.imshow(lattice, interpolation='nearest',cmap='magma')
	plt.grid()
	plt.show()
	return

@jit(nopython=True)
def DLA():
	L = 300
	steps = 1000
	particles = 1000
	lattice = np.zeros((L,L))
	moves = [np.array([0,1]),np.array([1,0]),np.array([0,-1]),np.array([-1,0])]
	lattice[L // 2,L // 2] = 1
	angles = np.random.uniform(0.0,2 * np.pi,(particles))
	R = 11.
	counter = 0
	ct = 0
	lattices = []
	for angle in angles:
		break_flag = False
		pos = np.array([int(L / 2 + R * np.cos(angle)),int(L / 2 + R * np.sin(angle))])
		while (not break_flag):
			for nb in moves:
				new_p = pos + nb
				if (lattice[new_p[0], new_p[1]] == 1):
					lattice[pos[0], pos[1]] = 1
					pos_loc = pos - np.array([L // 2,L // 2])
					dist = np.sqrt(np.sum(pos_loc ** 2)) + 10
					R = max(R, dist)
					counter += 1
					break_flag = True
					break
			pos += moves[random.randint(0,3)]
			pos_loc = pos - np.array([L // 2,L // 2])
			break_flag = break_flag or (np.sqrt(np.sum(pos_loc ** 2)) > R + 50) or (pos[0] > L - 1 or pos[0] < 1 or pos[1] > L - 1 or pos[1] < 1)
		if counter % 50 == 0:
			lattices.append(lattice.copy())
			#print(ct)
			#ct += 1
			#if ct > 5:
			#	break
	return lattices

@jit(nopython=True)
def DLA_1():
	lim = 150
	L = 1000
	N = 5000
	ret_lis = []
	particles = 10000
	lattice = np.zeros((L * L))
	moves = [1,L,-1,-L]
	lattice[L // 2 * (L + 1)] = 1
	angles = np.random.uniform(0.0,2 * np.pi,(particles))
	R = 11.
	counter = 1
	for angle in angles:
		pos_ = [int(L / 2 + R * np.cos(angle)), int(L / 2 + R * np.sin(angle))]
		pos = L * pos_[0] + pos_[1]
		choices = np.random.randint(0,4, size=N)
		mv_p = np.zeros((len(choices)))
		for i in range(len(choices)):
			mv_p[i] = moves[choices[i]]
		csmvp = np.cumsum(mv_p) + pos
		poss = np.transpose(np.stack([csmvp // L,csmvp % L])) - np.tile(np.array([L // 2,L // 2]), (N,1))
		dist = np.linalg.norm(poss,axis=1)
		dist_check = dist > R + lim
		lt = [np.take(lattice, csmvp + nb) for nb in moves]
		lt.append(dist_check)
		nbs = np.any(np.stack(lt), axis = 0)
		frt = np.argmax(nbs)
		if (dist_check[frt]):
			continue
		pos = csmvp[frt]
		for nb in moves:
			if (pos + nb >= 0 and pos + nb < L * L):
				if lattice[pos + nb]:
					lattice[pos] = 1
					counter += 1
					if counter % 50 == 0:
						temp = np.reshape(np.array(list(lattice)), (L,L))
						ret_lis.append(temp)
					if dist[frt] + 10 > R:
						R = dist[frt] + 10
			else:
				return ret_lis
	return ret_lis

def DLA_draw():
	try:
		os.mkdir('DLA')
	except Exception:
		pass
	try:
		os.mkdir('DLA/jit')
	except Exception:
		pass
	t1 = time.time()
	lts = DLA()
	t2 = time.time()
	print(t2 - t1)
	print(len(lts))
	for i in range(len(lts)):
		plt.clf()
		plt.imshow(lts[i], interpolation='nearest',cmap='magma')
		plt.grid()
		plt.savefig('DLA/jit/img' + str(i) + '.png')

@jit(nopython=True)
def sweep(grid, beta, L):
	pos_arr = np.random.randint(0, L, (L * L,2))
	probs = np.random.uniform(0.,1.,size = (L * L))
	for pos,r in zip(pos_arr,probs):
		delta = (grid[pos[0],(pos[1] + 1) % L] + grid[pos[0],(pos[1] - 1) % L] + grid[(pos[0] - 1) % L,pos[1]] + grid[(pos[0] + 1) % L,pos[1]]) * (2. * grid[pos[0], pos[1]])
		if delta < 0:
			grid[pos[0], pos[1]] *= -1
		else:
			prob = np.exp(-beta * delta)
			grid[pos[0], pos[1]] *= ((r >= prob) - (r < prob)) #check
def MCS(L=5, norm=1000):
	Ts = np.linspace(1.,5.,5)
	mag_mods = {}
	grid = np.ones((L,L))
	for B,T in zip((1. / Ts),Ts):
		mag_mods[T] = []
		for i in range(norm):
			sweep(grid, B, L)
		for i in range(5000):
			sweep(grid, B, L)
			mag_mods[T].append(np.sum(grid) / L ** 2)
	for T in Ts:
		print(T, np.mean(np.abs(mag_mods[T])))
	return

@jit(nopython=True)
def chi(grid, L, col_ind):
	sum = 0.
	for col_ind_loc in range(L):
		for row_ind_loc in range(L):
			sum += grid[row_ind_loc, col_ind_loc] * grid[row_ind_loc, (col_ind_loc + col_ind) % L]
	return sum / L ** 2

def MCS_hb(L=500):
	T = 2.
	mag_mods = {}
	grid = np.random.choice([-1,1],(L,L))
	ts = [10,20,50,100,200,500,1000,2000,5000]
	R = []
	for i in range(5001):
		sweep(grid, 1 / T, L)
		#if i in [10,100,1000,5000]:
		#	Cs = np.array([chi(grid, L, l) for l in range(L // 2)])
		#	lim = -(np.argmax(np.flip(Cs > np.exp(-1.))) + 1)
		#	Cs_trim = Cs[:lim]
		#	C = -np.sum(np.log(Cs_trim))
		#	plt.plot(Cs)
		#	#plt.imshow(grid, interpolation='nearest',cmap='magma')
		#	#plt.grid()
		#	plt.show()
		if i in ts:
			Cs = np.array([chi(grid, L, l) for l in range(L // 2)])
			lim = -(np.argmax(np.flip(Cs > np.exp(-1.))) + 1)
			Cs_trim = Cs[:lim]
			C = -np.sum(np.log(Cs_trim))
			lim += L
			R.append(lim * (lim + 1) / (2 * C))
		print(i)
	plt.clf()
	print(ts, R)
	plt.plot(ts,np.array(R) * (ts[0] ** (1 / 2)) / R[0],"*r",label="R")
	plt.plot(ts,np.sqrt(ts),"-g",label="t**1/2")
	plt.yscale("log")
	plt.xscale("log")
	plt.legend()
	plt.show()
	return

def test():
	ts = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
	R = 1 / np.array([76678.90254210735, 34257.08959153745, 15512.010142922078, 8395.911168268487, 6021.3671186718675, 4108.885033247207, 2868.6649283142388, 2087.453491472444, 1441.7309553167202])
	plt.plot(ts,R * (ts[0] ** (1 / 2)) / R[0])
	plt.plot(ts,ts ** (1 / 2))
	plt.yscale("log")
	plt.xscale("log")
	plt.show()

@jit(nopython=True)
def MCS_thr(B,L):
	grid = np.ones((L,L))
	mag_mods_loc = np.zeros((5000))
	for i in range(2000):
		sweep(grid, B, L)
	for i in range(5000):
		sweep(grid, B, L)
		mag_mods_loc[i] = np.sum(grid) / L ** 2
	return [np.mean(np.abs(mag_mods_loc)), np.var(mag_mods_loc) * B * L ** 2]

@jit(nopython=True)
def grid_energy(grid, L):
	energy = 0.
	for i in range(L):
		for j in range(L):
			energy -= grid[i,j] * (grid[(i - 1) % L,j] + grid[(i + 1) % L,j] + grid[i,(j - 1) % L] + grid[i,(j + 1) % L])
	#energy = -np.sum(grid * (np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0)
	#+ np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1)))
	return energy

def MCS_thr_2(B,L):
	grid = np.ones((L,L))
	energies = np.zeros((5000))
	for i in range(2000):
		sweep(grid, B, L)
	for i in range(5000):
		sweep(grid, B, L)
		energies[i] = grid_energy(grid, L)
	return np.var(energies) * (B / L) ** 2

Ts = np.linspace(1.,5.,1000)

def MCS_2(L=5):
	return list(map(list, zip(*Parallel(n_jobs=20)(delayed(MCS_thr)(B,L) for B in (1. / Ts)))))

def MCS_3(L):
	return Parallel(n_jobs=11)(delayed(MCS_thr_2)(B,L) for B in (1. / Ts))

def an_en(beta):
	k = 2 * np.tanh(2 * beta) / np.cosh(2 * beta)
	kp = 2 * np.tanh(2 * beta) ** 2 - 1
	return 2 / np.pi * (beta / np.tanh(2 * beta)) ** 2 * (2 * spec.ellipk(k ** 2) - 2 * spec.ellipe(k ** 2) - (1 - kp) * (np.pi / 2 + kp * spec.ellipk(k ** 2)))

def MCS_2_main():
	#lim = np.argmax(Ts > (2.  / (np.log(1 + np.sqrt(2)))))
	#Ts2 = Ts[:lim]
	#res = []
	#for L in [10]:#,20]:
	#	t1 = time.time()
	#	res.append(MCS_2(L))
	#	t2 = time.time()
	#	print(L, t2 - t1)
	MCS_hb()
	#test()
	#plt.plot(Ts,res[0][0],".b",label="L=10")
	#plt.plot(Ts,res[1][0],".g",label="L=20")
	#plt.plot(Ts2,(1-np.sinh(2/Ts2)**-4)**(1./8),"-r",label="analityczne")
	#print(res[0] / an_en(1.  / Ts))
	#plt.plot(Ts, res[0],"*g",label="numerycznie")
	#plt.plot(Ts, 4 * an_en(1.  / Ts),"-r",label="analityczne")
	#plt.legend()
	#plt.show()
	#plt.plot(Ts,res[0][1],".r",label="L=10")
	#plt.plot(Ts,res[1][1],".g",label="L=20")
	#plt.legend()
	#plt.show()
	#plt.imshow(lattice, interpolation='nearest',cmap='magma')
	#plt.grid()
	#plt.show()
	return

def lnp():
	M = np.loadtxt("C:\\Users\\26kub\\source\\repos\\skwf_cpp\\skwf_cpp\\data.txt")
	plt.plot(M[:,0],M[:,2],".r",label="L=10")
	plt.legend()
	plt.show()

def lnp2():
	M = np.loadtxt("C:\\Users\\26kub\\source\\repos\\SIR_simulation\\sim_data.txt")
	plt.plot(M[:,0],M[:,2],".r",label="L=10")
	plt.legend()
	plt.show()

def main():
	#DLA_draw()
	#MCS()
	#MCS_2_main()
	lnp2()
	return

main()