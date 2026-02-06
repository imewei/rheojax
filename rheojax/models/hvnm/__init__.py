"""HVNM (Hybrid Vitrimer Nanocomposite Model) package.

Constitutive models for nanoparticle-filled vitrimers with permanent,
exchangeable, dissociative, and interphase subnetworks.

Classes
-------
HVNMLocal
    Local (0D) Hybrid Vitrimer Nanocomposite Model
"""

from rheojax.models.hvnm.local import HVNMLocal

__all__ = ["HVNMLocal"]
