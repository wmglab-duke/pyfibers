: This channels is implemented by Jenny Tigerholm.
:The steady state curves are collected from Winkelman 2005
:The time constat is from Gold 1996 and Safron 1996
: To plot this model run KA_Winkelman.m
: Adopted and altered by Nathan Titus

NEURON {
	SUFFIX ka34
	USEION k READ ek WRITE ik
	RANGE gbar, ek, ik
	RANGE tau_m, minf, hinf,tau_h,m,h, gp, g
	RANGE minfshift, hinfshift, mtaushift, htaushift, ik

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
	mtaushift = 0 (ms)
	htaushift = 0 (ms)
}

ASSIGNED {
	v	(mV) : NEURON provides this
	ik	(mA/cm2)
	g	(S/cm2)
	tau_m	(ms)
    tau_h   (ms)
    minf
    hinf
	gp
    ek	(mV)
	celsius (degC)
}

STATE { h m }

BREAKPOINT {
	SOLVE states METHOD cnexp
	gp = m*m*m*h
	g = gbar*gp
	ik = g * (v-ek)
}

INITIAL {
	: assume that equilibrium has been reached
    rates(v)
	m=minf
    h=hinf

}

DERIVATIVE states {
	rates(v)
	m' = (minf - m)/(tau_m)
    h' = (hinf - h)/(tau_h)

}

? rates
PROCEDURE rates(Vm (mV)) (/ms) {
	LOCAL Q10
		TABLE minf,hinf,tau_m,tau_h DEPEND celsius FROM -120 TO 100 WITH 440
UNITSOFF
		Q10 = q10^((celsius-22)/10)
        :minf=1/(1+exp(-1*(Vm+16.8-minfshift)/20.7))
		:minf=1/(1+exp(-1*(Vm-28-minfshift)/25.3))
        :hinf=1/(1+exp((Vm+74-hinfshift)/14))
		:hinf=1/(1+exp((Vm+29-hinfshift)/8))
        :tau_m=6.09/(exp((Vm+33.64)/6.7)+exp(-1*(Vm+46.6)/14.71)) + 2.5/(1+exp((Vm-26.2)/21.55))
        :tau_m=0.66 + 1/(0.13*exp((Vm-mtaushift)/13.0) + 0.04*exp(-1*(Vm-mtaushift)/39.3))
		:tau_h= 1*(4 + 82.59/(exp((Vm+83.39)/9.046)^2 + exp(-1*(Vm+101.6)/30)) + 15/(1+exp((Vm+42.58)/20.79)))
		:tau_h=9+24.04*exp(-1*(Vm-htaushift)/23.2)
		:tau_h=10+33.51/(exp((Vm+0.03-htaushift)/12.6)+exp(-1*(Vm+128.7-htaushift)/7.824))

		minf = (1/(1+exp(-1*(Vm-24)/17)))^(1/3)
		hinf = 1/(1+exp((Vm+31)/12))
		tau_m = (1/(exp((Vm-39)/13)+exp(-1*(Vm+134)/47)))/2
		tau_h = 15 + 1/(exp((Vm-60)/15)+exp(-1*(Vm+300)/27)) :The vertical offset seems too large compared to current plots

        tau_m=tau_m/Q10
        tau_h=tau_h/Q10
UNITSON

}
