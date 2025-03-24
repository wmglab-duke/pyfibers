: This channel is implemented by Nathan Titus 2019.
: Data From:

NEURON {
	SUFFIX kv21
	USEION k READ ek WRITE ik
	RANGE gbar, ek, ik
	RANGE tau_m,minf,hinf,tau_h,m,h
	RANGE minfshift, hinfshift, ik, gp, g
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
}

ASSIGNED {
	v	(mV) : NEURON provides this
	ik	(mA/cm2)
	g	(S/cm2)
	tau_m	(ms)
    tau_h  (ms)
    minf
    hinf
	gp
    ek	(mV)
	celsius (degC)
}

STATE { h m }

BREAKPOINT {
	SOLVE states METHOD cnexp
	gp = m^3*h
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
		:TABLE minf,hinf,tau_m,tau_h DEPEND celsius FROM -120 TO 100 WITH 440
UNITSOFF
		Q10 = q10^((celsius-22)/10)
        minf= (1/(1+exp((-1*(Vm - 1)/10))))^(1/3)
        hinf= 0.15 + .85/(1+exp(((Vm + 27)/7)))
        tau_m = 2+1.2/(exp((Vm-15)/8.5)+exp(-1*(Vm+68)/15))
        tau_h = 45/(exp((Vm-10.2)/8)+exp(-1*(Vm+101)/8)) + 8200/(1+exp((-1*Vm/60)))

        tau_m=tau_m/Q10
        tau_h=tau_h/Q10
UNITSON

}
