#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 13:15:36 2018

Predictive information on u-skate pattern in Gray-Scott model

@author: manuelbaltieri
"""

from jpype import *
import random
import math
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

# check if JVM was started
def init_jvm(jvmpath=None):
    if isJVMStarted():
        return
    startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

# Change location of jar to match yours:
jarLocation = "../../infodynamics.jar"
# Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
#startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)
init_jvm()




# Create a PI calculator and run it:
piCalcClass = JPackage("infodynamics.measures.continuous.kraskov").PredictiveInfoCalculatorKraskov
piCalc = piCalcClass()
piCalc.setProperty("NORMALISE", "true") # Normalise the individual variables
piCalc.initialise(1) # Use history length 1 (Schreiber k=1)
piCalc.setProperty("k", "4") # Use Kraskov parameter K=4 for 4 nearest points
# Perform calculation with correlated source:
piCalc.setObservations(JArray(JDouble, 1)(sourceArray), JArray(JDouble, 1)(destArray))


result = piCalc.computeAverageLocalOfObservations()
# Note that the calculation is a random variable (because the generated
#  data is a set of random variables) - the result will be of the order
#  of what we expect, but not exactly equal to it; in fact, there will
#  be a large variance around it.
# Expected correlation is expected covariance / product of expected standard deviations:
#  (where square of destArray standard dev is sum of squares of std devs of
#  underlying distributions)
corr_expected = covariance / (1 * math.sqrt(covariance**2 + (1-covariance)**2));
print("TE result %.4f nats; expected to be close to %.4f nats for these correlated Gaussians" % \
    (result, -0.5 * math.log(1-corr_expected**2)))
# Perform calculation with uncorrelated source:
teCalc.initialise() # Initialise leaving the parameters the same
teCalc.setObservations(JArray(JDouble, 1)(sourceArray2), JArray(JDouble, 1)(destArray))
result2 = teCalc.computeAverageLocalOfObservations()
print("TE result %.4f nats; expected to be close to 0 nats for these uncorrelated Gaussians" % result2)