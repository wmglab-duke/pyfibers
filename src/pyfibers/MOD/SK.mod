: This channel is implemented by Nathan Titus 2019.
: Calcium-Activated potassium channel of BK and SK Channel Contributions
: Kv7.2 (n) and Kv7.3 (m). Kv7.5 was considered "similar enough" to
: Kv7.2 so as to be considered a part of that current (n).
: Data From:

NEURON {
	SUFFIX sk
	USEION k READ ek WRITE ik
	USEION ca READ cai
	RANGE gbar, ek, ik
	RANGE tau_m,ninf,tau_n,n
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
    tau_n  (ms)
	cai		(mM)
    ninf
	gp
    ek	(mV)
	celsius (degC)
}

STATE { n }

BREAKPOINT {
	SOLVE states METHOD cnexp
	gp = n
	g = gbar*gp
	ik = g * (v-ek)
}

INITIAL {
	: assume that equilibrium has been reached
    rates(v)
    n=ninf :SK

}

DERIVATIVE states {
	rates(v)
    n' = (ninf - n)/(tau_n)

}

? rates
PROCEDURE rates(Vm (mV)) (/ms) {
	LOCAL Q10,sf,v12,pca
UNITSOFF
		pca = log10(cai)-3
		Q10 = q10^((celsius-22)/10)
        ninf= 1/(1+exp(-1*(pca + 6.4)/.12))
        tau_n = -1*pca :I was unable to find a value for this, but it should depend on concentration and be fairly fast


        tau_n=tau_n/Q10
UNITSON

}
