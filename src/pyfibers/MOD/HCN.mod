: h.mod is the h channel
: from Tom Andersson Sensitivity studies of voltage-dpendent conductance in neurons
: Tom has build his model on Kouranova 2008 Hyoerpolarization -activated cyclic nuleotide-gated channel mRNA and protein expression in large versus mall diameter dorsal root ganglion neurons: correlation with hyperpolarization-activated current
: Adopted and altered by Nathan Titus (small changes in relative ik/ina and reversal potentials

NEURON {
	SUFFIX hcn
	USEION k READ ek, ko, ki WRITE ik
        USEION na READ ena, nao, nai WRITE ina
	RANGE gbar, ena, ik, ina, ek, ekna, g_s, g_f, ih
	RANGE minf, ninf, m, n, gp, g, q10, tau_n, tau_m
}

UNITS {
	(molar) = (1/liter)			: moles do not appear in units
	(mM)	= (millimolar)
	(S) = (siemens)
	(mV) = (millivolts)
	(mA) = (milliamp)
}

PARAMETER {
	gbar	(S/cm2): = 30e-6
	ekna = -30   (mV): the combined rev pot from Na and k
	:g_s	= 0.67
	:g_f	= 0.33
	g_s = .25
	g_f = .75
	q10 = 3

}

ASSIGNED {
	v	(mV) : NEURON provides this
	ik	(mA/cm2)
    ina     (mA/cm2)
	ih  (mA/cm2)
	g	(S/cm2)
	gp

	tau_m	(ms)
    tau_n (ms)
	ninf
    minf
    :    kh
	ek	(mV)
	ena	(mV)
	ki	(mM)
	ko	(mM)
	nai	(mM)
	nao	(mM)
	celsius (degC)
}

STATE { m n }

BREAKPOINT {
	SOLVE states METHOD cnexp
	gp = (g_s*n*n*n+g_f*m*m*m)
	g = gbar * gp
    ina=(ekna-ek)*g*(v-ena)/(ena-ek)
    ik=(ekna-ena)*g*(v-ek)/(ek-ena)
	ih = ina+ik
    : This math looks complicated but it is just parallel
	: conductance model math to set the reversal potential
	: of the channel to ekna based on the relative proportion
	: of ik and ina
}

INITIAL {
	: assume that equilibrium has been reached
	rates(v)
	m = minf
    n = ninf

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

	minf = (1/(1+exp((Vm+97)/7.35)))^(1/3)
    ninf = (1/(1+exp((Vm+94)/8.9)))^(1/3)

    tau_m = 0.5/(exp((Vm-42)/11.8)+exp(-1*(Vm+498)/66.6))

    tau_n= 0.5/(exp((Vm+25)/4.1)+exp(-1*(Vm+356)/32))

	tau_m=tau_m/Q10
    tau_n=tau_n/Q10
UNITSON


}
