from solvers.IRISSolver import iris
from solvers.IRISToolSolver import iris_tool
from solvers.OldIRISSolver import old_iris
from solvers.other import t

c = True
while c:
    print("1. IRIS cod propriu")
    print("2. IRIS cu tool")
    print("3. IRIS vechi")
    inp = int(input("Alege : "))
    if inp == 1:
          iris()
    elif inp == 2:
        iris_tool()
    elif inp == 3:
        old_iris()
    elif inp == 4:
        t()
    else:
        c = False