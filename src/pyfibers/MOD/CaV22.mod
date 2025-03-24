: N-Type Voltage Dependent Calcium Channel NOT FOR VAGAL AFFERENTS

: Coded and modified by Nathan Titus 2018
: Development Notes: Almost definitely a second inactivation gate or
: concentration dependency. Insufficient data exists so inactivation tau
: should be updated when more is understood about these channels.

NEURON {
	SUFFIX cav22
	USEION ca READ eca WRITE ica
	RANGE ica
	RANGE gbar, minf, mtau, hinf, htau, sinf, stau
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
	sinf
	stau
	ica (mA/cm2)
	eca (mV)
}


INITIAL {
	rates(v)
	m = minf
	h = hinf
	s = sinf
}

STATE {
	m h s
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	gp = m*m*m*h*s
	g = gbar*gp
	ica = g*(v-eca)

}

DERIVATIVE states {
	rates(v)
	m' = (minf - m)/(mtau)
	h' = (hinf - h)/(htau)
	s' = (sinf - s)/(stau)
}

? rates
PROCEDURE rates(Vm (mV)) {
	LOCAL Q10
	TABLE minf,hinf,sinf,mtau,htau,stau DEPEND celsius FROM -120 TO 100 WITH 440

UNITSOFF
	Q10 = q10^((celsius-22)/10)
	:minf = 1/(1 + exp(-1*(Vm - 1.5)/6))
	minf = (1/(1 + exp(-1*(Vm + 4)/7.5)))^(1/3)
	:hinf = 1/(1 + exp((Vm + 61.5)/12))
	hinf = 1/(1 + exp((Vm + 48)/7))
	sinf = 1/(1 + exp((Vm + 81)/8.6))
	mtau = .1 + 0.5/(exp((Vm-3)/6.7)+exp(-1*(Vm+37)/13.5))
	htau = 36/(exp((Vm-35)/15.4)+exp(-1*(Vm+134)/26.6)) + 19 + 50/(1+exp((Vm+50)/10))
	htau = 10/(exp((Vm-54)/23)+exp(-1*(Vm+150)/35))
	stau = 50 + 30/(exp((Vm-50)/26)+exp(-1*(Vm+150)/26))

	mtau = mtau/Q10
	htau = htau/Q10
	stau = stau/Q10

}

UNITSON
