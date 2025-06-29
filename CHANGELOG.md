# CHANGELOG

<!-- version list -->

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
