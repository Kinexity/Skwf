import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from matplotlib.patches import Circle
import imageio
import scipy.constants as const

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
	return 4 * ((1 / d) ** 12 - (1 / d) ** 6)

def particles():
	particleRow = 4
	particleNumber = particleRow ** 2
	boxsize = 8.0
	V = boxsize ** 2
	eps = 1.0
	sigma = 1.0
	promien = 0.5
	dt = 0.0001
	temp = 2.5
	kB = 1
	dx = boxsize / particleRow
	delta = 1e-15
	T = 10
	r_c = 2.5
	N_t = int(T / dt)
	plot_step = N_t / int(T / 1e-1)
	plotting = True
	lin = np.tile(np.linspace(0,particleRow - 1, particleRow), (particleRow,1))
	pos = np.transpose(np.reshape(np.stack([lin, np.transpose(lin)]), (2,particleNumber))) * dx + dx / 2
	v = np.random.random((particleNumber,2)) - 0.5
	pos_tiled = np.tile(pos, (particleNumber, 1, 1))
	T_arr = []
	p_arr = []
	k_b = const.Boltzmann
	K_0 = np.sum(v ** 2) / (particleNumber * k_b)
	cycles = N_t
	for tp in range(N_t):
		pos_tiled = np.tile(pos, (particleNumber, 1, 1))
		temp_r_ij = pos_tiled - np.transpose(pos_tiled, axes=(1,0,2))
		t_abs = np.abs(temp_r_ij)
		t_sgn = np.sign(temp_r_ij)
		diff = - t_sgn * boxsize + temp_r_ij
		r_ij = np.where(t_abs < 4, temp_r_ij, diff)
		d = np.linalg.norm(r_ij, axis = 2) + delta
		norm_r_ij = np.stack([r_ij[:,:,0] / d, r_ij[:,:,1] / d])
		F_init = f(d) - f(r_c)
		F_mag = np.where(d <= r_c, F_init,0.)
		F = norm_r_ij * F_mag
		F_s = np.transpose(np.sum(F, axis=1))
		v += F_s * dt
		pos += v * dt
		pos %= boxsize
		T = np.sum(v ** 2) / (2 * particleNumber * k_b)
		np.fill_diagonal(F_mag,0.)
		P = particleNumber * k_b * T / V - np.sum(F_mag * d) / (2 * V)
		if (T > 10 * K_0):
			cycles = tp
			break
		T_arr.append(T)
		p_arr.append(P)
		if (tp % plot_step == 0 and plotting): # co 100-na klatka
			plt.clf() # wyczyść obrazek
			F = plt.gcf() # zdefiniuj nowy
			nStr = str(tp)
			nStr = nStr.rjust(5,'0')
			for r in pos: # pętla po cząstkach
				a = plt.gca() # ‘get current axes’ (to add smth to them)
				cir = Circle(((r[0] + 4) % 8, (r[1] + 4) % 8), radius=promien) # kółko tam gdzie jest cząstka
				a.add_patch(cir) # dodaj to kółko do rysunku
				plt.plot() # narysuj
			plt.xlim((0,boxsize)) # obszar do narysowania
			plt.ylim((0,boxsize))
			F.set_size_inches((6,6)) # rozmiar rysunku
			plt.title("Symulacja gazu Lennarda-Jonesa, krok " + nStr)
			plt.savefig('img' + nStr + '.png')
	images = []
	if plotting:
		for tp in range(cycles):
			if (tp % plot_step == 0):
				nStr = str(tp) #nagraj na dysk – numer pliku z 5 cyframi, na początku zera, np 00324.png
				nStr = nStr.rjust(5,'0')
				images.append(imageio.imread('img' + nStr + '.png'))
		imageio.mimsave('movie.gif', images)#, duration=10)
	plt.clf()
	plt.plot(T_arr)
	plt.show()
	plt.plot(p_arr)
	plt.show()
	return

def main():
	particles()
	return

main()