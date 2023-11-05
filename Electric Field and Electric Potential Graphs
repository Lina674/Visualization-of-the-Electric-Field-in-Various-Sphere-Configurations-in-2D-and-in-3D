import matplotlib.pyplot as plt
from numpy import linspace
from scipy.optimize import minimize_scalar

#r=20
#q=30


choice=input('Sphere, concentric sphere, or shell (s/sc/hs)?')
if choice=="s":

  choice1=input('Conductor or insulator (c/i)?')
  if choice1=="c":
    r=float(input('Enter a radius (m)'))
    q=float(input('Enter a uniformly distributed charge Q (C)'))
    #maxE=(9*(10**9))*q*(r**-2)
    #maxV=(9*(10**9))*q*(r**-1)
    def fn(x):
      if (x <= r):
        return 0
      elif (x <= 7*r):
        return (9*(10**9))*q*(x**-2)

    xList = linspace(0, 9999, 5000)
    yList = [fn(x) for x in xList]

    def fn(v):
      if (v <= r):
        return (9*(10**9))*q*(r**-1)
      if (v <= 7*r):
        return (9*(10**9))*q*(v**-1)

    vList = linspace(0, 9999, 5000)
    zList = [fn(v) for v in vList]


  if choice1=="i":
    r=float(input('Enter a radius (m)'))
    q=float(input('Enter a uniformly distributed charge Q (C)'))

    def fn(x):
      if (x <= r):
        return (9*(10**9))*q*(x)*(r**-3)
      if (x <= 7*r):
        return (9*(10**9))*q*(x**-2)
    xList = linspace(0, 9999, 5000)
    yList = [fn(x) for x in xList]

    def fn(v):
      if (v <= r):
        return ((9*(10**9))*q*(-v**2)*(0.5)*(r**-3))+(3*0.5*(9*(10**9))*q*(r**-1))
      if (v <= 7*r):
        return (9*(10**9))*q*(v**-1)

    vList = linspace(0, 9999, 5000)
    zList = [fn(v) for v in vList]



elif choice=="sc":
  choice2=input('Inner shell (c/i)')
  if choice2=="i":
    ra=float(input('Enter a radius [a] (m)'))
    qa=float(input('Enter a uniformly distributed charge Q (C)'))
    rb=float(input('Enter a radius [b] (m)'))
    rc=float(input('Enter a radius [c] (m)'))
    qb=float(input('Enter a uniformly distributed charge Q (C)'))

    choice3=input('Outer shell (c/i)')
    if choice3=="c":
      def fn(x):
        if (x <= ra):
          return (9*(10**9))*qa*x*(ra**-3)
        if (x <= rb):
          return (9*(10**9))*qa*(x**-2)
        if (x <= rc):
          return 0
        if (x<= 3*rc):
          return (9*(10**9))*(qa+qb)*(x**-2)
      xList = linspace(0, 9999, 5000)
      yList = [fn(x) for x in xList]

      def fn(v):
        if (v <= ra):
          return ((9*(10**9))*qa*(-v**2)*(0.5)*(ra**-3))+(3*0.5*(9*(10**9))*qa*(ra**-1))
        if (v <= rb):
          return (9*(10**9))*qa*(v**-1)
        if (v<=rc):
          return (9*(10**9))*(qa+qb)*(rb**-1)
        if (v <= 3*rc):
          return (9*(10**9))*(qa+qb)*(v**-1) #not working
      vList = linspace(0, 9999, 5000)
      zList = [fn(v) for v in vList]

  if choice2=="c":
    ra=float(input('Enter a radius [a] (m)'))
    qa=float(input('Enter a uniformly distributed charge Q (C)'))
    rb=float(input('Enter a radius [b] (m)'))
    rc=float(input('Enter a radius [c] (m)'))
    qb=float(input('Enter a uniformly distributed charge Q (C)'))

    choice3=input('Outer shell (c/i)')
    if choice3=="c":
      def fn(x):
        if (x <= ra):
          return 0
        if (x <= rb):
          return (9*(10**9))*qa*(x**-2)
        if (x <= rc):
          return 0
        if (x<= 3*rc):
          return (9*(10**9))*(qa+qb)*(x**-2)
      xList = linspace(0, 9999, 5000)
      yList = [fn(x) for x in xList]

      def fn(v):
        if (v <= ra):
          return (9*(10**9))*qa*(ra**-1)
        if (v <= rb):
          return (9*(10**9))*qa*(v**-1)
        if (v<=rc):
          return (9*(10**9))*(qa)*(rb**-1)
        if (v <= 3*rc):
          return (9*(10**9))*(qa+qb)*(v**-1) #not working

      vList = linspace(0, 9999, 5000)
      zList = [fn(v) for v in vList]

    if choice3=="i":
      def fn(x):
        if (x <= ra):
          return 0
        if (x <= rb):
          return (9*(10**9))*qa*(x**-2)
        if (x <= rc):
          return ((9*(10**9))*qb*(x**3-rb**3)*((x**2)*(rc**3-rb**3)**-1))+(9*(10**9))*qa*(x**-2)
        if (x <= 5*rc):
          return ((9*(10**9))*(qb)*(x**-2)) #not working

      xList = linspace(0, 9999, 5000)
      yList = [fn(x) for x in xList]

      def fn(v):
        if (v <= ra):
          return ((9*(10**9))*qa*(-v**2)*(0.5)*(ra**-3))+(3*0.5*(9*(10**9))*qa*(ra**-1))
        if (v <= rb):
          return (9*(10**9))*qa*(v**-1)
        if (v<=rc):
          return (9*(10**9))*(qa+qb)*(rb**-1)
        if (v <= 3*rc):
          return (9*(10**9))*(qa+qb)*(v**-1) #not working

elif choice=="hs":
  choice1=input('Conductor or insulator (c/i)?')
  if choice1=="c":
    rin=float(input('Enter an inner radius (m)'))
    rout=float(input('Enter an outer radius (m)'))
    q=float(input('Enter a uniformly distributed charge Q (C)'))

    def fn(x):
      if (x <= rin):
        return 0
      if (x <= rout):
        return 0
      if (x <= 5*rout):
        return (9*(10**9))*q*(x**-2)

    xList = linspace(0, 9999, 5000)
    yList = [fn(x) for x in xList]

    def fn(v):
      if (v <= rin):
        return (9*(10**9))*q*(rin**-1)

      if (v <= 5*rout):
        return (9*(10**9))*q*(v**-1)

    vList = linspace(0, 9999, 5000)
    zList = [fn(v) for v in vList]

  if choice1=="i":
    rin=float(input('Enter an inner radius (m)'))
    rout=float(input('Enter an outer radius (m)'))
    q=float(input('Enter a uniformly distributed charge Q (C)'))

    def fn(x):
      if (x <= rin):
        return 0
      if (x<=rout):
        return ((9*(10**9))*q*(x**3-rin**3)*(((x**2)*(rout**3-rin**3))**-1))
      if (x <= 7*rout):
        return (9*(10**9))*q*(x**-2)
    xList = linspace(0, 9999, 5000)
    yList = [fn(x) for x in xList]

    def fn(v):
      if (v <= rin):
        return (9*(10**9))*q*(rin**-1)
      if (v <= rout):
        return -(9*(10**9))*q*((rout**3-rin**3)**-1)*(v**3+2*rin**3)*((2*v)**-1)+(9*(10**9))*q*(rout**-1)
      if (v <= 7*rout):
        return (9*(10**9))*q*(v**-1) #not working

    vList = linspace(0, 9999, 5000)
    zList = [fn(v) for v in vList]

plt.plot(xList, yList)
plt.xlabel("Distance from the Center (m)")
plt.ylabel("Electric Field (N/C)")
plt.title("Electric Field as a Function of Distance")

#plt.xlim(left=0)
plt.show()

plt.plot(vList, zList)
plt.xlabel("Distance from the Center (m)")
plt.ylabel("Electric Potential (V)")
#plt.xlim(left=0)
plt.title("Electric Potential as a Function of Distance")

plt.show()

#PRINT MAX VALUES AND EQUATIONS!!!
