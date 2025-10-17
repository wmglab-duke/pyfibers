# CHANGELOG

<!-- version list -->

## v0.8.0 (2025-10-16)

### Build System

- Update build pipeline to use semantic release publishing
  ([`d00b32a`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/d00b32af85ea40d57ab929ebb871b7ef762bf4fc))

### Documentation

- Add logging documentation
  ([`b464ca4`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/b464ca47e3fe675f6e091799b4bfe9294db9b175))

- Autodoc will not eval default function args
  ([`b464ca4`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/b464ca47e3fe675f6e091799b4bfe9294db9b175))

- Fix use of list in plotting tutorials
  ([`b464ca4`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/b464ca47e3fe675f6e091799b4bfe9294db9b175))

- Update block tutorial to run faster
  ([`b464ca4`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/b464ca47e3fe675f6e091799b4bfe9294db9b175))

### Features

- Replace print statements with logging and expose user control of logging
  ([`b464ca4`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/b464ca47e3fe675f6e091799b4bfe9294db9b175))


## v0.7.0 (2025-09-26)

### Documentation

- Add docs for sweeney model
  ([`556a486`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/556a4868ac050a2e9ad47a35148cc7dee5fe91ba))

### Features

- Add sweeney model to library
  ([`556a486`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/556a4868ac050a2e9ad47a35148cc7dee5fe91ba))


## v0.6.2 (2025-09-22)

### Bug Fixes

- Fix bug where intrastim could error from steady state simulation encroaching on t>0
  ([`adc8a3c`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/adc8a3c1079cec0d9f71148b26d6f3d5a7ea2a62))


## v0.6.1 (2025-09-12)

### Bug Fixes

- Fix bug where repeated creation of fiber using the same varname could cause fiber.time to stop
  recording
  ([`119ac12`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/119ac1253305fd07d87ea9097db9ba51c853fe57))

### Build System

- Add twine to ci/cd
  ([`807d784`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/807d78463a40ef6a0196b7ffa6a352106ac84e2e))


## v0.6.0 (2025-09-12)

### Bug Fixes

- **IntraStim**: Add cleanup routine so that IntraStim does not leave vestigial trainIClamp on
  fibers
  ([`f43d7ec`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/f43d7eceab1dae985f01ca9f55b5007b97c6d369))

Closes #423

- **IntraStim**: Add cleanup routine so that IntraStim does not leave vestigial...
  ([`f43d7ec`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/f43d7eceab1dae985f01ca9f55b5007b97c6d369))

### Documentation

- Add feature details to README
  ([`6cec1ae`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/6cec1ae0283e91662e54d23713481773af4f3a0e))

Closes #410

- Update documentation for fiber shifting
  ([`ec6c49b`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/ec6c49b8480315e2ba322bcae972ef9d2ba39d4c))

- Update README installation instructions
  ([`6cec1ae`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/6cec1ae0283e91662e54d23713481773af4f3a0e))

- Update README installation instructions
  ([`6cec1ae`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/6cec1ae0283e91662e54d23713481773af4f3a0e))

Closes #420

### Features

- Add the ability to shift coordinates for resample potentials
  ([`ec6c49b`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/ec6c49b8480315e2ba322bcae972ef9d2ba39d4c))


## v0.5.0 (2025-09-06)

### Build System

- Add force push to github push
  ([`e11cf79`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/e11cf79f721361db93e8a0d2b68abae07b7bb190))

- Added file to skip github jekyll
  ([`ed3b67f`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/ed3b67f2a36bfcf8b024bb3635ea4a372822d5ec))

- Change push target to correct link
  ([`3970931`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/3970931b17f51d547635aafb167a7fce176ee53d))

- Drop twine upload from gitlab-ci
  ([`a7d7a25`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/a7d7a25b69de03079ea80aa941adbb5c61a57aca))

### Documentation

- Update fiber model custom docs for new API
  ([`a64aff6`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/a64aff609b7740d22c945b923ccd1d8bdaf64e96))

### Features

- Update API for adding a custom fiber model
  ([`a64aff6`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/a64aff609b7740d22c945b923ccd1d8bdaf64e96))


## v0.4.2 (2025-06-29)

### Bug Fixes

- Move twine upload after github push
  ([`e7db12a`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/e7db12ae3f46c4f6c4c9c51b729c9a81908a20e1))


## v0.4.1 (2025-06-29)

### Bug Fixes

- Releases should now function properly
  ([`4d3ab8a`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/4d3ab8a1feda2352a5429275f45cc76175507a3c))

### Documentation

- Elie review pass on documentation
  ([`361b863`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/361b86382ac3e19f7057fc22c321716213e54751))

- Update latex engine and add proper version acquisition for docs
  ([`2fdab8f`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/2fdab8f24d887a868bd638a6f7ee04aa2f6db9c5))

- Update version for semantic release in docs version
  ([`d2f70dd`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/d2f70dde3d9a3ef88fe35e261aa2b2c00d1e4560))


## v0.4.0 (2025-03-24)

### Bug Fixes

- Added checks for incorrectly parameterized rest potential and for oscillations in vm during steady
  state simulation
  ([`a6dcf88`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/a6dcf883aaf7e497c0b2d2e120cee45350c108d4))

- Lowered default steady state time step to avoid oscillations in vm
  ([`a6dcf88`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/a6dcf883aaf7e497c0b2d2e120cee45350c108d4))

- Use negative absolute value of provided value for steady state time
  ([`a6dcf88`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/a6dcf883aaf7e497c0b2d2e120cee45350c108d4))

### Features

- Added fiber models from Thio 2024 publication
  ([`a6dcf88`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/a6dcf883aaf7e497c0b2d2e120cee45350c108d4))


## v0.3.3 (2025-03-21)

### Bug Fixes

- Change SCHILD model resting potential to correct value
  ([`a760b91`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/a760b911c0e8c98e453695c3bf5d4ca2bab71d2d))


## v0.3.2 (2025-03-21)

### Bug Fixes

- Add option to enforce odd node count in fiber creation
  ([`048de88`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/048de889fe3049aa53c0b6cd4278a3baf18d0ea2))

Also fixed a bug in stimulation where intracellular stimulation of passive nodes was only detected
  if passive_end_nodes = 1

Closes #377 Closes #349

### Documentation

- Update line length for all docs to 88
  ([`2078dce`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/2078dce85548a9e19c31371932ca7b6541795c3c))


## v0.3.1 (2025-03-07)

### Bug Fixes

- Added commitizen to pre-commit
  ([`fdd34cd`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/fdd34cdbe0d875bc029361706d8f5a31b59d2acd))

### Build System

- Add empty changelog
  ([`13e7278`](https://gitlab.oit.duke.edu/wmglab/wmglab-neuron/-/commit/13e72784bb2a17fb6349bc77d350f949b56324ef))


## v0.3.0 (2025-03-05)


## v0.2.1 (2025-02-17)


## v0.1.4 (2024-12-26)


## v0.1.1 (2024-07-18)


## v0.1.0 (2024-07-08)


## v0.0.5 (2024-06-10)


## v0.0.4 (2024-05-14)


## v0.0.3 (2024-04-25)


## v0.0.2 (2023-08-02)


## v0.0.1 (2023-05-23)
