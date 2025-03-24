: This channels is implemented by Jenny Tigerholm.
:The steady state curves are collected from Winkelman 2005
:The time constat is from Gold 1996 and Safron 1996
: To plot this model run KA_Winkelman.m
: Adopted and altered by Nathan Titus

NEURON {
	SUFFIX newnav8
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
	tau_m	(ms)
    tau_h   (ms)
	tau_s
    minf
    hinf
	sinf
	gp
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
        :minf = 1/(1+exp(-1*(Vm+3.3)/5.7))
		:hinf = 1/(1+exp((Vm+44)/6.5))
		:sinf = 1/(1+exp((Vm+32)/4.9))
		minf = (1/(1+exp(-1*(Vm+4)/7)))^(1/3)
		hinf = 1/(1+exp((Vm+28)/4.56))
		sinf = 1/(1+exp((Vm+50)/7.5))
		tau_m = 0.03 + .5/(exp((Vm+0)/12)+exp(-1*(Vm+29)/18))
		:tau_h = 2.3+7.9/(exp((Vm+28)/17.6)+exp(-1*(Vm+54)/23.3))
		tau_h = 2.63+250/(exp((Vm+36)/7.7)+exp(-1*(Vm+0)/16.3)) + 1.4/(1+exp(-1*(Vm+0.6)/2.95))
		tau_s = 34/(exp((Vm+2)/16)+exp(-1*(Vm+108)/8)) + 160/(1+exp(-1*(Vm+110)/75))

        tau_m=tau_m/Q10/2
        tau_h=tau_h/Q10
        tau_s=tau_s/Q10
UNITSON
}
