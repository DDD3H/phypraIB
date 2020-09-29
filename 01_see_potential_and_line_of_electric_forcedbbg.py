import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import numpy as np
from matplotlib import cm
from scipy.integrate import ode as ode
from itertools import product
def colorline(
        x, y, v, cmap='copper', norm=plt.Normalize(-3.0, 3.0),
        linewidth=0.5, alpha=1.0):

    x,y,v = np.array(x),np.array(y),np.array(v)
    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=v, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

class charge:
    def __init__(self, q, pos):
        self.q=q
        self.pos=pos
 
def E_point_charge(q, a, x, y):
    return q*(x-a[0])/((x-a[0])**2+(y-a[1])**2)**(1.5), \
        q*(y-a[1])/((x-a[0])**2+(y-a[1])**2)**(1.5)
 
def E_total(x, y, charges):
    Ex, Ey=0, 0
    for C in charges:
        E=E_point_charge(C.q, C.pos, x, y)
        Ex=Ex+E[0]
        Ey=Ey+E[1]
    return [ Ex, Ey ]

def E_norm(x,y, charges):
    Ex, Ey = E_total(x, y, charges)
    return np.sqrt(Ex**2+Ey*Ey)

def E_dir(t, y, charges):
    Ex, Ey=E_total(y[0], y[1], charges)
    n=np.sqrt(Ex**2+Ey*Ey)
    return [Ex/n, Ey/n]

def V_point_charge(q, a, x, y):
    return q/((x-a[0])**2+(y-a[1])**2)**(0.5)

def V_total(x, y, charges):
    V=0
    for C in charges:
        Vp=V_point_charge(C.q, C.pos, x, y)
        V = V+Vp
    return V

# charges and positions
# You can define
charges=[ charge(1, [-1, 0]), charge(1, [1, 0])]
 
# calculate field lines
x0, x1=-3, 3
y0, y1=-2.5, 2.5
R=0.01
# loop over all charges
xs,ys = [],[]
es = []
vs = []
for C in charges:
    # plot field lines starting in current charge
    dt=0.8*R
    if C.q<0:
        dt=-dt
    # loop over field lines starting in different directions 
    # around current charge
    for alpha in np.linspace(0, 2*np.pi*31/32, 32):
        r=ode(E_dir)
        r.set_integrator('vode')
        r.set_f_params(charges)
        x=[ C.pos[0] + np.cos(alpha)*R ]
        y=[ C.pos[1] + np.sin(alpha)*R ]
        e=[ E_norm(x[0],y[0],charges) ]
        v=[ V_total(x[0],y[0],charges)]
        r.set_initial_value([x[0], y[0]], 0)
        while r.successful():
            r.integrate(r.t+dt)
            x.append(r.y[0])
            y.append(r.y[1])
            e.append(E_norm(r.y[0],r.y[1],charges))
            v.append(V_total(r.y[0],r.y[1],charges))
            hit_charge=False
            # check if field line left drwaing area or ends in some charge
            for C2 in charges:
                if np.sqrt((r.y[0]-C2.pos[0])**2+(r.y[1]-C2.pos[1])**2)<R:
                    hit_charge=True
            if hit_charge or (not (x0<r.y[0] and r.y[0]<x1)) or \
                    (not (y0<r.y[1] and r.y[1]<y1)):
                break
        xs.append(x)
        ys.append(y)
        es.append(e)
        vs.append(v)


# calculate electric potential
vvs = []
xxs = []
yys = []
numcalcv = 300
for xx,yy in product(np.linspace(x0,x1,numcalcv),np.linspace(y0,y1,numcalcv)):
    xxs.append(xx)
    yys.append(yy)
    vvs.append(V_total(xx,yy,charges))
xxs = np.array(xxs)
yys = np.array(yys)
vvs = np.array(vvs)

fig, ax = plt.subplots(facecolor="w",figsize=(6,5))

for x,y,v in zip(xs,ys,vs):
    lc = colorline(x, y, v, cmap='jet',linewidth=0.75)
cbar = plt.colorbar(lc)
cbar.set_label("Electric Potential and Line of Electric Force")

clim0,clim1 = -2,2
vvs[np.where(vvs<clim0)] = clim0*0.999999 # to avoid error
vvs[np.where(vvs>clim1)] = clim1*0.999999 # to avoid error
plt.tricontour(xxs,yys,vvs,10,colors="0.3",linewidths=0.75)

plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)

ax.set_aspect("equal")
plt.savefig("electric_force_lines_change_lc_1.png",dpi=250,bbox_inches="tight",pad_inches=0.02)
plt.show()
