: This channel is implemented by Nathan Titus 2019.
: Calcium-Activated potassium channel of BK and SK Channel Contributions
: Kv7.2 (n) and Kv7.3 (m). Kv7.5 was considered "similar enough" to
: Kv7.2 so as to be considered a part of that current (n).
: Data From:

NEURON {
	SUFFIX bk
	USEION k READ ek WRITE ik
	USEION ca READ cai
	RANGE gbar, ek, ik
	RANGE tau_m,minf,tau_n,m,h,hinf,tau_h
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
	cai		(mM)
    minf
	hinf
	gp
    ek	(mV)
	celsius (degC)
}

STATE { m h }

BREAKPOINT {
	SOLVE states METHOD cnexp
	gp = m*h
	g = gbar*gp
	ik = g * (v-ek)
}

INITIAL {
	: assume that equilibrium has been reached
    rates(v)
	m=minf :BK
    h=hinf :BK

}

DERIVATIVE states {
	rates(v)
	m' = (minf - m)/(tau_m)
	h' = (hinf - h)/(tau_h)

}

? rates
PROCEDURE rates(Vm (mV)) (/ms) {
	LOCAL Q10,sf,v12,vh12,pca
UNITSOFF
		Q10 = q10^((celsius-22)/10)
		pca = log10(cai)-3 :converts to log10(cai [molar])
		v12 =  -50*pca-232
		vh12 =  -8*pca+35
        minf= 1/(1+exp(-1*(Vm - v12)/24))
        hinf= 1/(1+exp((Vm - vh12)/47))
		tau_m = 1/(exp((Vm+(58*pca)+303)/(3.2*pca))+exp(-1*(Vm+(107*pca)+453)/(6.8*pca)))+0.4
		tau_h = 1/(exp((Vm+(3*pca)+100)/(3*pca))+exp(-1*(Vm+(191*pca)+600)/(17*pca)))


        tau_m=tau_m/Q10
        tau_h=tau_h/Q10
UNITSON

}
