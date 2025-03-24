: This channel is implemented by Nathan Titus 2019.
: M-type potassium channel which directly represents the contributions of
: Kv7.2 (n) and Kv7.3 (m). Kv7.5 was considered "similar enough" to
: Kv7.2 so as to be considered a part of that current (n).
: Data From:

NEURON {
	SUFFIX km
	USEION k READ ek WRITE ik
	RANGE gbar, ek, ik
	RANGE tau_m,minf,ninf,tau_n,m,h
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
    tau_n  (ms)
    minf
    ninf
	gp
    ek	(mV)
	celsius (degC)
}

STATE { n m }

BREAKPOINT {
	SOLVE states METHOD cnexp
	gp = 0.25*m^3+0.75*n^3
	g = gbar*gp
	ik = g * (v-ek)
}

INITIAL {
	: assume that equilibrium has been reached
    rates(v)
	m=minf :fast
    n=ninf :slow

}

DERIVATIVE states {
	rates(v)
	m' = (minf - m)/(tau_m)
    n' = (ninf - n)/(tau_n)

}

? rates
PROCEDURE rates(Vm (mV)) (/ms) {
	LOCAL Q10
		TABLE minf,ninf,tau_m,tau_n DEPEND celsius FROM -120 TO 100 WITH 440
UNITSOFF
		Q10 = q10^((celsius-22)/10)
        minf= (1/(1+exp((-1*(Vm + 44)/6.4))))^(1/3)
        ninf= (1/(1+exp((-1*(Vm + 32)/9.2))))^(1/3)
        :tau_m = 7.7/(exp((Vm-58.3)/26.9)+exp(-1*(Vm+46.2)/8)) + 15/(1+exp((-1*(Vm + 87.5)/7.9)))
		tau_n = 15 + 62/(exp((Vm-13)/20)+exp(-1*(Vm+90)/20)) + 50/(1+exp((-1*(Vm + 50)/8)))
        :tau_n = 28/(exp((Vm-8.4)/12)+exp(-1*(Vm+76)/13.3)) + 80
		tau_m = 104/(exp((Vm-7)/20)+exp(-1*(Vm+32)/20)) + 30/(1+exp((-1*(Vm + 40)/80)))

        tau_m=tau_m/Q10/2
        tau_n=tau_n/Q10/2
UNITSON

}
