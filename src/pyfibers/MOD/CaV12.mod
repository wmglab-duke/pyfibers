: L-Type Voltage Dependent Calcium Channel

: Coded and modified by Nathan Titus 2018
: Development Notes:
:
:

NEURON {
	SUFFIX cav12
	USEION ca READ eca WRITE ica
	RANGE ica
	RANGE gbar, minf, mtau, hinf, htau
	RANGE q10, gp, g
}

UNITS {
    (molar) = (1/liter)                     : moles do not appear in units
    (mM)    = (millimolar)
	(uA) = (microamp)
	(mA) = (milliamp)
	(mV) = (millivolt)
	(um) = (micron)
	(S) = (siemens)

}

PARAMETER {
	gbar = 6.5e-9 (S/cm2) :from Schild et al 1994
	q10 = 3
}

ASSIGNED {
	v (mV)
	celsius (degC)
	g (S/cm2)
	gp
	minf
	mtau
	hinf
	htau
	ica (mA/cm2)
	eca (mV)
}


INITIAL {
	rates(v)
	m = minf
	h = hinf
}

STATE {
	m h
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	gp = m*m*m*h
	g = gbar*gp
	ica = g*(v-eca)

}

DERIVATIVE states {
	rates(v)
	m' = (minf - m)/(mtau)
	h' = (hinf - h)/(htau)
}

? rates
PROCEDURE rates(Vm (mV)) {
	LOCAL Q10
	TABLE minf,hinf,mtau,htau DEPEND celsius FROM -120 TO 100 WITH 440

UNITSOFF
	Q10 = q10^((celsius-22)/10)
	:minf = 1/(1 + exp(-1*(Vm - 1.5)/6))
	minf = (1/(1 + exp(-1*(Vm - 1.75)/10)))^(1/3)
	:hinf = 1/(1 + exp((Vm + 61.5)/12))
	hinf = 1/(1 + exp((Vm - 10)/8))
	:mtau = 0.57 + 5.5/(exp((Vm + 4.6)/7.95) + exp(-1*(Vm - 8.9)/33.8))
	mtau = 0.25 + 0.5/(exp((Vm - 15)/5) + exp(-1*(Vm + 27)/12))
	:htau = 4.3 + 167/(exp((Vm + 15)/9.2) + exp(-1*(Vm + 38)/31)) + 80/(1+exp((Vm + 72)/5))
	htau = 1.4/(exp((Vm - 145)/30) + exp(-1*(Vm + 150)/16.4))
	mtau = mtau/Q10
	htau = htau/Q10

}
UNITSON
