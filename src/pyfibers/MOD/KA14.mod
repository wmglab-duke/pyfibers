: This channels is implemented by Jenny Tigerholm.
:The steady state curves are collected from Winkelman 2005
:The time constat is from Gold 1996 and Safron 1996
: To plot this model run KA_Winkelman.m
: Adopted and altered by Nathan Titus

NEURON {
	SUFFIX ka14
	USEION k READ ek WRITE ik
	RANGE gbar, ek, ik
	RANGE tau_m,minf,hinf,tau_h,sinf,tau_s,m,h,s
	RANGE minfshift, hinfshift, sinfshift, ik, gp, g
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
}

ASSIGNED {
	v	(mV) : NEURON provides this
	ik	(mA/cm2)
	g	(S/cm2)
	tau_m	(ms)
    tau_h  (ms)
	tau_s  (ms)
    minf
    hinf
	sinf
	gp
    ek	(mV)
	celsius (degC)
}

STATE { h s m }

BREAKPOINT {
	SOLVE states METHOD cnexp
	gp = m*m*m*h*s
	g = gbar*gp
	ik = g * (v-ek)
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
PROCEDURE rates(Vm (mV)) (/ms) {
	LOCAL Q10
		TABLE minf,hinf,sinf,tau_m,tau_h,tau_s DEPEND celsius FROM -120 TO 100 WITH 440
UNITSOFF
		Q10 = q10^((celsius-22)/10)
        minf=(1/(1+exp(-1*(Vm+25)/12)))^(1/3)
        hinf=.073 + 0.924/(1+exp((Vm+47)/4.75))
        sinf = .415 + .576/(1+exp((Vm+44.5)/5.93)) + .103/(1+exp(-1*(Vm)/18.37))
        tau_m = (.6 + 2748/(exp((Vm+128)/14.5)+exp(-1*(Vm+10)/8))+1.7/(1+exp((Vm+8.5)/10.65)))/2
        tau_h = 35 + 11.22/(exp((Vm+21.4)/9.48)+exp(-1*(Vm+155.3)/16.4))
		:tau_h = 64 + 92.3/(exp((Vm+44.3)/8.74)+exp(-1*(Vm+107)/17))
		:tau_h = 56 + 218.5/(exp((Vm+51.9)/9.48)+exp(-1*(Vm+153.3)/17.7))
		tau_s = 1000*(3.97/(exp((Vm+35.1)/9.97)+exp(-1*(Vm+83.3)/18))+2.5/(1+exp(-1*(Vm+27.3)/7.11)))

        tau_m=tau_m/Q10
        tau_h=tau_h/Q10
        tau_s=tau_s/Q10
UNITSON

}
