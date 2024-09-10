# Implementations of Fiber Models
In PyFibers, we have implemented a number of models describing the ultrastructure and membrane properties of both myelinated and unmyelinated peripheral nerve fibers. Herein we list the available fiber models, and note any changes from the original publications. Unless otherwise specified, the model code was adapted from ASCENT \cite{musselman_ascent_2021}

## Myelinated Fiber Models
We provide three variants of the Mcintyre-Richarson-Grill (MRG) model \cite{McIntyre2002} for myelinated fibers. The MRG model is a biophysically detailed model of a myelinated nerve fiber, and is widely used in the field of computational neuroscience. The three variants are:
- `MRG-discrete`: The original MRG model, as described in \cite{McIntyre2002}.
- `MRG-interpolation`: An interpolation of the MRG model, which allows for modeling of any fiber diameter between 2 and 16 microns, as described in \cite{musselman_ascent_2021}.
- `Small MRG-interpolation`: A modification of MRG parameters to better replicate the behavior of thinly myelinated fibers, as described in \cite{pena_cap_2024}.

## Unmyelinated Fiber Models
We provide several variants of unmyelinated C-fiber models.
- `Tigerholm`: A biophysically detailed model of a C-fiber, as described in \cite{Tigerholm2018}.
- `Rattay`: A simplified version of the Tigerholm model, as described in \cite{Rattay2019}.
- `Schild 1994`: A simple model of a C-fiber, as described in \cite{Schild1994}.
- `Schild 1997`: A modification of the Schild 1994 model, as described in \cite{Schild1997}.
- `Sundt`: A modification of the Schild 1994 model, as described in \cite{Sundt2015}.

## Passive end nodes
When constructing a fiber, we allow users to specify the number of passive end nodes at each end of the fiber. In this case, the fiber is constructed as normal. For end nodes designated as passive, all mechanisms other than extracellular ('extracellular') are removed from the node. Then, the following properties are set as described in \cite{Pelot2021}:
- The passive ('pas') mechanism is added to the node.
- The membrane capacitance is set to 1 uF/cm^2. (typical value for membrane)
- The membrane resistance is set to 0.0001 S/cm^2. (slightly lower than typical value for membrane, dulls current entry into the node)
- The passive reversal potential is set to the resting potential of the fiber.
- The axial resistivity is set to 1e10 Ohm*cm. (effectively infinite, so no current can flow axially. Avoids the creation of action potentials through restricting flow of current to the next node)
