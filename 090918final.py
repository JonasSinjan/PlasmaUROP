import numpy as np
from scipy.constants import m_e, eV, m_p
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D

T_e = 1 * eV
v_te_sqr = 2 * T_e / m_e
cs = np.sqrt(eV / m_p)  # cold bohm speed for 1eV electrons
ne = 10 ** 16
ni = 10 ** 16  # ne = ni


def ionthermalv_sqr(T):
    return 2 * T / m_p


def electronthermalv_sqrt(T):
    return 2 * T / m_e


def fe(v, theta, Te):  # integrated over azimuth and changed coordinates
    return 2 / (np.pi ** 0.5 * (electronthermalv_sqrt(Te)) ** (3 / 2)) * (v ** 2) * np.exp(
        -(v ** 2) / electronthermalv_sqrt(Te)) * np.sin(theta)


def fivector(v, theta, T, vi):  # integrated over azimuth and changed coordinates
    return 2 / (np.pi ** 0.5 * (ionthermalv_sqr(T)) ** (3 / 2)) * v ** 2 * np.sin(theta) * \
           np.exp(-((v ** 2) - 2 * v * np.cos(theta) * vi + vi ** 2) / ionthermalv_sqr(T))


def lhs(v, theta, T, vi):
    return fivector(v, theta, T, vi) * v * np.cos(theta)  # as in temps i did v not vz


def rhs(v, theta, T, vi, Te):
    return 1 / 3 * fivector(v, theta, T, vi) * ((v ** 2) - 2 * v * np.cos(theta) * vi + vi ** 2) + 1 / 3 * m_e / m_p * (
        v) ** 2 * fe(v, theta, Te)


def lhsval(T, vi):
    return dblquad(lhs, 0, np.pi, lambda v: 0, lambda v: 100 * cs, args=(T, vi,))


def rhsval(T, vi, Te, lim):  # around 300cs should do it
    return dblquad(rhs, 0, np.pi, lambda v: 0, lambda v: lim * cs, args=(T, vi, Te,))


def tempion(v, theta, T, vi):
    return (1 / 3) * m_p * ((v ** 2) - 2 * v * np.cos(theta) * vi + vi ** 2) * fivector(v, theta, T, vi)


def electrontemp(v, theta, Te):
    return (1 / 3) * m_e * (v) ** 2 * fe(v, theta, Te)


def plot(datapoints):
    templist = sp.linspace(0, 5 * eV, datapoints)
    tstart = templist[0] / eV
    tend = templist[-1] / eV
    lhslist = []
    rhslist = []
    yerrlist = []
    Te = eV
    vi = 10 * cs
    lim = 300
    for y in range(datapoints):
        T = templist[y]

       # print(T / eV)
        etemp = dblquad(electrontemp, 0, np.pi, lambda v: 0, lambda v: lim * cs, args=(Te,))
        itemp = dblquad(tempion, 0, np.pi, lambda v: 0, lambda v: (lim / 5) * cs,
                        args=(T, vi,))  # checking single integrals and adding to the overall rhs function
       # print(itemp[0] / eV)
        altrhs = np.sqrt((etemp[0] + itemp[0]) / m_p) / cs
        lhslist.append(lhsval(T, vi)[0] / cs)
        yval, err = rhsval(T, vi, Te, lim)
        x = np.sqrt(yval) / cs
        rhslist.append(x)
        print(x, altrhs)  # where the comparison is printed out
        yerrlist.append((1 / 2) * err / cs)
        if T==0:
            rhslist[0]=1#the integral fails when T=0 as you are dividing by zero in the function
    print(lhslist)
    print(yerrlist)

    etemplist = [t / eV for t in templist]
    plt.figure(0)
    plt.errorbar(etemplist, rhslist, yerrlist, xerr=None, fmt='o', label='Integrals')
    newy = np.sqrt((Te + templist) / m_p)  # plotting it with errors
    plt.plot(templist / eV, newy / cs, 'r-', label='$\gamma$ = 1')
    threey = np.sqrt((Te + 3 * templist) / m_p)  # plotting gamma of 3
    plt.plot(templist / eV, threey / cs, 'b-', label="$\gamma$ = 3")
    fivethirdsy = np.sqrt((Te + 5 / 3 * templist) / m_p)  # plotting gamma of 5/3
    plt.plot(templist / eV, fivethirdsy / cs, 'g-', label="$\gamma$ = 5/3")
    plt.legend()
    plt.title(f"Flowing Ions from {tstart} - {tend} eV, fixed drift at {int(vi/cs)} c_s, electrons at 1eV")
    plt.xlabel("Ion temperature (eV)")
    plt.ylabel("Minimum velocity at the sheath edge (units of c_s)")
    plt.show()


plot(20)

# function to plot 3D ion temperature - fixed temp, so expect a flat horizontal plane
# dont go too far on int limit, max 200*cs
def iontemplot3d(length, Temp, driftstart, driftend, limstart, limend):
    T = Temp * eV  # setting ion temp
    iontempmat = np.zeros((length, length))
    iontemplist = []
    errlist = []
    ioncounter = 0  # these counters are all to see if the results are within a range from the true result
    vmax = sp.linspace(limstart * cs, limend * cs,
                       length)  # 10-200
    driftlist = sp.linspace(driftstart * cs, driftend * cs, length)  # 0-10
    csdriftlist = [i / cs for i in driftlist]  # normalising to cs (cold bohm speed-electrons at 1eV)
    X, Y = np.meshgrid(vmax / cs, csdriftlist)
    for x in range(length):  # varying both the drift velocity and the upper limit on v in the integral
        v_i = driftlist[x]
        for i in range(length):
            vlim = vmax[i]
            tempcheck = dblquad(tempion, 0, np.pi, lambda v: 0, lambda v: vlim, args=(T, v_i,))
            if tempcheck[0] >= (T - 0.05 * T) and tempcheck[0] <= (T + 0.05 * T):  # within 0.1 percent
                ioncounter += 1
                print(tempcheck[0] / eV, T / eV, tempcheck[1] / T, vlim / cs,
                      v_i / cs, )  # the errors become massive at high drift
            iontempmat[x][i] = tempcheck[0] / eV
            iontemplist.append(tempcheck[0] / eV)
            errlist.append(tempcheck[1] / eV)
    fig = plt.figure(1)  # for 3D iontemp varying drift vel and vel limit
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, iontempmat)
    print(errlist)
    print(ioncounter, 100 * ioncounter / (length * length))
    plt.title(f"{T/eV}eV Ion temperature")
    plt.xlabel("V limit on integral (units of c_s)")
    plt.ylabel("Ion Drift velocity (units of c_s)")
    ax.set_zlabel("Numerical Solution to Integral (eV)")
    plt.show()
    # plt.figure(0)    #for iontemp varying int limit 2D
    # plt.plot(vmax/cs,iontemplist)
    # plt.title(f"{T/eV}eV temperature")
    # plt.xlabel("V limit on integral (units of c_s)")
    # plt.ylabel("Temperature Calculated (eV)")


#iontemplot3d(50,5,0,10,10,200)

# plot of T_integral/T_function - ideally straight line at 1
def Tcomparisonplot(datapoints):
    vlist = sp.linspace(0, 50 * cs, datapoints)
    T = 5 * eV
    baalrudlist = []
    for i in range(datapoints):
        v_i = vlist[i]
        intlimit = 300  # very dependent on limit, above 200 starts to oscillate around 1, 100 works best
        Iontemp = dblquad(tempion, 0, np.pi, lambda v: 0, lambda v: intlimit * cs, args=(T, v_i,))
        baalrudlist.append(Iontemp[0] / T)

    plt.plot(vlist / cs, baalrudlist)
    plt.title(f"T_Integral/T_Ion for varying drift velocity, integral limit = {intlimit}*c_s ")
    plt.xlabel("Drift Velocity (units of c_s)")
    plt.ylabel("T_Integral/T_Ion")
    plt.ylim(0, 1.5)
    plt.show()


# Tcomparisonplot(200)
# also checked with drift of 50 and limit of 50, tailed off as the limit means part of the maxwellian is missed out

# now plotting 3D velocity check, varying the drift and limit on the integral, should expect a flat plane at 45degrees to the horizontal, y=x so to speak
def velcheckplot3D(length, driftstart, driftend, limstart,
                   limend, T):  # ylength always same as xlength-hence just length for size of square matrix:velmat
    velcounter = 0
    velmat = np.zeros((length, length))
    driftlist = sp.linspace(driftstart * cs, driftend * cs, length)
    vmax = sp.linspace(limstart * cs, limend * cs, length)
    # vellist = []
    for x in range(length):  # varying both the drift velocity and the upper limit on v in the integral
        v_i = driftlist[x]
        for i in range(length):
            vlim = vmax[i]
            velcheck = dblquad(lhs, 0, np.pi, lambda v: 0, lambda v: vlim, args=(T, v_i,))
            velmat[x][i] = (velcheck[0] / cs)
            if v_i == 0 and velcheck[0] <= 1e-10:  # to ensure correct results for zero arent included in the velcounter
                print(velcheck[0] / cs)
                continue
            elif velcheck[0] <= (v_i - 0.05 * v_i) or velcheck[0] >= (v_i + 0.05 * v_i):
                print(velcheck[0] / cs, v_i / cs, np.abs(100 * (velcheck[0] - v_i) / v_i), velcheck[1] / velcheck[0],
                      vlim / cs)
                velcounter += 1  # number of points more than 5% from true value
    # plt.figure(1)  #for velocity int varying limit 2D-doesnt work when using nested for loops, as can only vary the integral limit, not the actual drift
    # plt.plot(vmax/cs, vellist, 'r-')
    # plt.xlabel("V limit on integral (units of c_s)")
    # plt.ylabel("Velocity calculated (units of c_s)")
    # plt.title(f"{vi/cs} (units of c_s)")
    fig2 = plt.figure(2)  # varying drift and v limit of int for velocity int, should be independent of drift
    ax = fig2.add_subplot(111, projection='3d')
    csdriftlist = [i / cs for i in driftlist]  # normalising to cs (cold bohm speed-electrons at 1eV)
    X, Y = np.meshgrid(vmax / cs, csdriftlist)
    ax.plot_surface(X, Y, velmat)
    print(velcounter, length * length,
          100 * velcounter / (length * length))  # velcounter is how many points out 5% band
    plt.title("Fluid Flow Velocity Integral")
    plt.xlabel("V limit on integral (units of c_s)")
    plt.ylabel("Ion Drift velocity (units of c_s)")
    ax.set_zlabel("Calculated Integral (units of c_s)")
    plt.show()


#velcheckplot3D(20, 0, 10, 10, 200, 5*eV)  # must not start integral limit at zero, or division by zero error will occur

def electrontempplot3d(length, Tstart, Tend, Tpoints, limstart, limend):
    electroncounter = 0
    Tlist = sp.linspace(Tstart * eV, Tend * eV, Tpoints)  # 1-2#100 points
    electrontempmat = np.zeros((len(Tlist), length))
    vmax = sp.linspace(limstart * cs, limend * cs, length)  # 10-400
    for z in range(len(Tlist)):  # for 3D electron temp, varying temp and int limit
        Tel = Tlist[z]
        for i in range(length):
            vlim = vmax[i]
            tempelectroncheck = dblquad(electrontemp, 0, np.pi, lambda v: 0, lambda v: vlim, args=(Tel,))
            err = 0.05  # change this when considering different ranges
            if tempelectroncheck[0] / eV >= (Tel / eV - err * Tel / eV) and tempelectroncheck[0] / eV <= (
                    Tel / eV + err * Tel / eV):  # within 5 percent
                print(Tel / eV, tempelectroncheck[0] / eV)
                electroncounter += 1
                print(f"The error is {tempelectroncheck[1]/eV} eV")  # want to see what errors the method is estimating
            electrontempmat[z][i] = (tempelectroncheck[0] / eV)
    fig4 = plt.figure(4)  # 3D plot for electron temp, varying temp and v limit on int
    ax = fig4.add_subplot(111, projection='3d')
    T, Y = np.meshgrid(vmax / cs, Tlist / eV)
    ax.plot_surface(T, Y, electrontempmat)
    print(electroncounter)
    plt.title("Temperature Integral")
    plt.xlabel("V limit on integral (units of c_s)")
    plt.ylabel("Electron Temperature (eV)")
    ax.set_zlim(0, Tend)
    plt.show()
    # plt.figure(3)   #2D plot of electron temp varying the v limit on integral#doesnt work for nested for loops
    # plt.plot(vmax / cs, electrontemplist)
    # plt.title(f"Electron Temperature at {T_e/eV} eV")
    # plt.xlabel("V limit on integral (units of c_s)")
    # plt.ylabel(f"Electron temperature calculated")

#electrontempplot3d(10,1,2,100,10,400)
