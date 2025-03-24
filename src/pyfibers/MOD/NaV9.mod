: This channels is implemented by Jenny Tigerholm.
:The steady state curves are collected from Winkelman 2005
:The time constat is from Gold 1996 and Safron 1996
: To plot this model run KA_Winkelman.m
: Adopted and altered by Nathan Titus

NEURON {
	SUFFIX nav9
	USEION na READ ena WRITE ina
	RANGE gbar, ena, ina
	RANGE tau_m, minf, hinf,tau_h, sinf, tau_s, m,h,s
	RANGE minfshift, hinfshift, mtaushift, htaushift, ina
	RANGE sinfshift, staushift, gp, g
}

UNITS {
	(S) = (siemens)
	(mV) = (millivolts)
	(mA) = (milliamp)
}

PARAMETER {
	gbar 	(S/cm2)
    q10 = 3
    minfshift = 0 (mV)
	hinfshift = 0 (mV)
	sinfshift = 0 (mV)
	mtaushift = 0 (ms)
	htaushift = 0 (ms)
	staushift = 0 (ms)
}

ASSIGNED {
	v	(mV) : NEURON provides this
	ina	(mA/cm2)
	g	(S/cm2)
	gp
	tau_m	(ms)
    tau_h   (ms)
	tau_s
    minf
    hinf
	sinf
    ena	(mV)
	celsius (degC)
}

STATE { h m s}

BREAKPOINT {
	SOLVE states METHOD cnexp
	gp = m*m*m*h*s
	g = gbar*gp
	ina = g * (v-ena)
}

INITIAL {
	: assume that equilibrium has been reached
    rates(v)
	m=minf
    h=hinf
	s=sinf

}

DERIVATIVE states {
	rates(v)
	m' = (minf - m)/(tau_m)
    h' = (hinf - h)/(tau_h)
	s' = (sinf - s)/(tau_s)

}

? rates
PROCEDURE rates(Vm (mV)) {
	LOCAL Q10
	TABLE minf,hinf,sinf,tau_m,tau_h,tau_s DEPEND celsius FROM -120 TO 100 WITH 440

UNITSOFF
		Q10 = q10^((celsius-22)/10)
        minf = (1/(1+exp(-1*(Vm+51)/8.4)))^(1/3)
		hinf = 1/(1+exp((Vm+55)/11.5))
		sinf = 0.03 + 0.97/(1+exp((Vm+79)/6.87))
		tau_m = (0.3 + 1/(exp((Vm+1.3)/22)+exp(-1*(Vm+82)/10)))/2
		tau_h = 2 + 1/(exp((Vm-32)/14)+exp(-1*(Vm+157)/12))
		tau_s = 1/(exp((Vm-195)/29)+exp(-1*(Vm+193)/12.4))+1165/(1+exp(-1*(Vm+40)/30))

        tau_m=tau_m/Q10
        tau_h=tau_h/Q10
        tau_s=tau_s/Q10
UNITSON
}
