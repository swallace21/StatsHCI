import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import statshci as st

a1 = [10,22,34,44,54,115,99,400]
a2 = [1,4,5,7,5,481,90,134,0,0,1,2,3,4]
st.shapiro_wilks(a1,a2)
