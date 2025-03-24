TITLE hh.mod   squid sodium, potassium, and leak channels

COMMENT

ENDCOMMENT

UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
	(S) = (siemens)
}

? interface
NEURON {
        SUFFIX nav7
        USEION na READ ena WRITE ina
        RANGE gbar, ina, gp, g
        RANGE minf, hinf, htau, mtau
	THREADSAFE : assigned GLOBALs will be per thread
}

PARAMETER {
        gbar = .12 (S/cm2)	<0,1e9>
		q10 = 3
}

STATE {
        m h
}

ASSIGNED {
        v (mV)
        celsius (degC)
        ena (mV)
		gp
		g (S/cm2)
        ina (mA/cm2)
        minf hinf
		mtau (ms)
		htau (ms)
}

? currents
BREAKPOINT {
        SOLVE states METHOD cnexp
		gp = m*m*m*h
        g = gbar*gp
	ina = g*(v - ena)
}


INITIAL {
	rates(v)
	m = minf
	h = hinf
}

? states
DERIVATIVE states {
        rates(v)
        m' =  (minf-m)/(mtau)
        h' = (hinf-h)/(htau)
}

:LOCAL q10


? rates
PROCEDURE rates(Vm(mV)) {  :Computes rate and other constants at current v.

	LOCAL Q10             :Call once from HOC to initialize inf at resting v.
        TABLE minf, mtau, hinf, htau DEPEND celsius FROM -120 TO 100 WITH 200

UNITSOFF
        Q10 = q10^((celsius - 21)/10)
    :"m" sodium activation system
			:mtau = 0.0178 + 100/(exp(-1*(Vm+18)/9) + exp((Vm-13)/15))/q10
			:mtau = 1/(exp((Vm+23.7)/12.35) + exp(-1*(Vm+13.6)/17))
			:mtau = 0.002185/(exp((Vm-42.4)/13.3)+exp(-1*(Vm+103.5)/13.5)) + .06951/(1+exp(-1*(Vm+81)/31.7))
			mtau = 0.07/(exp((Vm-11.4)/14)+exp(-1*(Vm+61)/9.4)) + .1/(1+exp(-1*(Vm+5.5)/8))
			minf = (1/(1+exp(-1*(Vm+25)/7)))^(1/3)
    :"h" sodium inactivation system
			htau = 1/(exp(-1*(Vm+131)/9.5) + exp((Vm+5.7)/12.4))
			hinf = 1/(1+exp((Vm+79)/7))
		mtau = mtau/Q10/2
		htau = htau/Q10
}

UNITSON
