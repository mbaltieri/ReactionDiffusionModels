# Reaction Diffusion Models

The Gray-Scott model represents one of the most studied autocatalytic systems (a system where the product of a chemical reaction is also a catalyst for the same reaction). This model has been previously proposed as a testing ground for theories of the origins of life, as shown in work by Nathaniel Virgo and colleagues in their studies on the origins of life.

This repository contains code for the simulation of the Gray-Scott system, generating a variety of [patterns](https://www.youtube.com/watch?v=ypYFUGiR51c&ab_channel=RobertMunafo) extensively reported in [computational studies](https://groups.csail.mit.edu/mac/projects/amorphous/GrayScott/). In particular, the code focuses on the creation of '[u-skate](https://mrob.com/pub/comp/xmorphia/uskate-world.html)' shapes as one of the simplest self-propulsive patterns within this system. 

Using standard measures from information theory (transfer entropy and predictive information), this code reports preliminary attempts to study the emergence of self-organising patters (u-skates) that are persistent to different types of perturbations. As one should expect however, the task is made daunting by a few different key problems, including:
- choice of corse-graining
- non-stationarity of the transition in autocatalytic reactions giving rise to the u-skate.'

For more information, see this old [blogpost](http://eon.elsi.jp/information-and-regulation-at-the-origins-of-life/).
