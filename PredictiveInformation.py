#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 16:06:30 2018

Test Predictive information

@author: manuelbaltieri
"""

from jpype import *
import numpy
import matplotlib.pyplot as plt
# Our python data file readers are a bit of a hack, python users will do better on this:
sys.path.append("/Users/manuelbaltieri/Dropbox/Python/ReactionDIffusion/demos/python")

plt.close('all')

def readFloatsFile(filename):
	"Read a 2D array of floats from a given file"
	with open(filename) as f:
		# Space separate numbers, one time step per line, each column is a variable
		array = []
		for line in f:
			# read all lines
			if (line.startswith("%") or line.startswith("#")):
				# Assume this is a comment line
				continue
			if (len(line.split()) == 0):
				# Line is empty
				continue
			array.append([float(x) for x in line.split()])
		return array


# Add JIDT jar library to the path
jarLocation = "/Users/manuelbaltieri/Dropbox/Python/ReactionDIffusion/infodynamics.jar"
# Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
def init_jvm(jvmpath=None):
    if isJVMStarted():
        return
    startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)
    
init_jvm()

# 0. Load/prepare the data:
dataRaw = readFloatsFile("/Users/manuelbaltieri/Dropbox/Python/ReactionDIffusion/v25_history.txt")
# As numpy array:
data = numpy.array(dataRaw)
# 1. Construct the calculator:
calcClass = JPackage("infodynamics.measures.continuous.kraskov").ActiveInfoStorageCalculatorKraskov
calc = calcClass()
# 2. Set any properties to non-default values:
# No properties were set to non-default values

calc.setProperty("k", "1")
calc.setProperty("AUTO_EMBED_RAGWITZ_NUM_NNS", "4")

result = numpy.zeros((50))

# Compute for all variables:
for v in range(50):
    # For each variable:
    variable = data[:, v]

    # 3. Initialise the calculator for (re-)use:
    calc.initialise()
    # 4. Supply the sample data:
    calc.setObservations(variable)
    # 5. Compute the estimate:
    result[v] = calc.computeAverageLocalOfObservations()

    print("AIS_Kraskov (KSG)(col_%d) = %.4f nats" %
        (v, result[v]))

plt.figure()
plt.plot(result)
plt.show()