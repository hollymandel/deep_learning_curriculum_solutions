This repo contains solutions to some of the exercises in Jacob Hilton's <a href = "https://github.com/jacobhilton/deep_learning_curriculum">Deep Learning Curriculum.</a>

<b>1-transformers</b>: a pytorch implementation of a decoder-only transformer 
for sequence tasks in `transformer.py`. It successfully completes a sequence reversal 
task (on the second half of the sequence) in `transformer_reverser.ipynb`. In 
`transformer-shakespeare.ipynb` I train it on the complete works of Shakespeare.

<b> 2-scaling</b>: replicating scaling law results using convolutional nets of
different sizes on the MNIST dataset. The figure at the bottom of `MNIST_scaling.ipynb`
is a replication is Figure 2 in <a href = "https://arxiv.org/pdf/2001.08361">Kaplan et al.</a> 

<b>3-parallelization</b>: an implementation of data parallelization using MPI in `data_parallel.py`.
A comparison of this parallel training and sequential training of CNNs on MNIST in `compare_parallel_single.ipynb`.

<b>6-RL</b>: trained a Cartpole agent using naive policy gradient in `policy_gradient_cartpole.ipynb`. 
Wrote a pytorch implementation of PPO in `ppo.py` and used it to trian a Lunar Lander agent in 
`ppo_lunar_lander.ipynb`.
