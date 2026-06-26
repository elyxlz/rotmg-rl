# An Ultra Opinionated Guide to Reinforcement Learning

> By Joseph Suarez (PufferLib / PufferAI). Saved here as the canonical RL reference for this
> project. Source: https://x.com/jsuarez5341/status/1943692998975402064 (and puffer.ai).
> This is the formalized version of how Joseph trains new PufferLib contributors.

Reinforcement learning is about learning through interaction. Applications include robotics,
logistics, gaming, and even control problems in science like nuclear fusion. It's an
underexplored niche of AI where you can really advance the field without a ton of compute. But
learning RL is hard, and most of the material out there for beginners makes it even harder. The
advice here is a formalized version of how I train new PufferLib contributors. Some of these came
in with zero programming knowledge and now help advance our research and tools. The key is to
start doing reinforcement learning immediately while filling in knowledge gaps slowly through
experience. In other words, reinforcement learning.

First, the content here is a radical departure from all other RL books, tutorials, and
references. This field is broken. Let me say that again: this field is broken. Mainstream deep
learning has high-quality resources, generally sane approaches to most common problems, and a
pipeline to start quickly doing useful work. The standard advice for newcomers in RL may as well
just be designed to screw with you. You will understand why by the end of this.

Second, this guide assumes that you are a competent programmer, have at least seen C or C++ in a
systems course somewhere, and have basic deep learning knowledge at the level of Stanford's
CS231n. My definition of "competent" includes many undergrads and excludes many well-paid ML
engineers. As a quick litmus test: can you implement the forward pass of an LSTM without importing
anything? If not or if the implementation you just thought of would be >100 lines, start with the
prequel article "My Advice for Programming and ML."

And finally, since I'm writing this for a broader audience: hi, I'm Joseph. After finishing my PhD
at MIT, I've started a quest to make RL fast and sane. Most of my content focuses on the cutting
edge of RL, but this has become surprisingly accessible lately. If you haven't heard of it
already, PufferLib is my main project. It's a reinforcement learning library that is both simpler
and 1000x faster than other libraries like SB3, RLlib, etc. We're always looking for new
contributors, so follow me here and join discord.gg/puffer to get involved. The best way to
support my work for free in <5 seconds is to star pufferlib on GitHub and RT this article.

## 1. Build a Simple Environment and Train an Agent

Read the introduction to my RL Quickstart Guide. Just the first paragraph. You now know some words
we use in RL and not much else.

Read the PufferLib docs on writing your own environment. Including all the linked code on the
squared sample environment. Map the terms from the introduction to this code. Observations,
actions, rewards, and terminals are just arrays. The observations are inputs to the agent's neural
network, which outputs actions. Rewards determine whether the agent has reached its goal, at which
point the terminal value is set to true. You don't know how we use all these things to actually
train yet and that is fine.

Write your own environment. Keep it so simple as to not even be useful. That will come next. If
you can't think of anything, do a stripped down version of flappy bird on a 2-block tall grid. The
agent can move either up or down. It observes whether there is a wall on the roof or floor. -1
reward for hitting the ceiling, 0 otherwise. Bind it to PufferLib following the docs and train your
first agent. Ask in the Discord if you get stuck.

## 2. Learn the most Basic Fundamentals

Read Karpathy's policy gradient blog. Including the implementation. There are a few very minor
details that are dated, but it is otherwise still the best basic introduction to on-policy methods.
You should now understand how policy gradients turns observations, actions, and rewards into
derivatives over weights. You have also seen discounted reward as your first example of an
advantage function. Understand that the discount factor mathematically determines how much you
care about reward now vs. reward later.

Read the Fundamentals section of my RL quickstart guide. This will give you context for the basic
classes of methods. Don't go down a rabbit hole of reading papers just yet. All modern algorithms
are at least slightly off-policy, and the strict distinction does not matter as much anymore. Read
the multi-agent bullet point twice.

Read and train Puffer Target. This environment is included in the same tutorial you already
followed. It is multi-agent and significantly more complex than squared. The agents no longer move
on the grid and have to solve a more temporally extended problem. Every agent sees a normalized
distance measure from itself to every other agent and goal. Agents only receive a sparse reward of
1 for reaching their goal. Follow the docs to train an agent on this environment. It should only
take a few seconds.

## 3. Build a Slightly More Complex Environment

Make something about as complex as Target. Max ~300 lines. It doesn't have to be multi-agent. If
you did grid-based flappybird before, do the full game. Any similarly scoped project should
suffice. Think about what data the agent needs to see in order to play and make sure that is
included in the observations. Reread the debugging section of the custom environment docs to avoid
common errors and post in our Discord if you get stuck. Aim to train an agent that you can visually
confirm is playing the game well.

Read other puffer environments. Snake and Convert are two slightly more complex environments with
clean code. You can also start looking at some of the arcade games like pong and breakout. By now,
you've probably hit a few bugs and can start to appreciate the ways in which things can go wrong.
We mitigate that by keeping code simple and minimally abstracted. The biggest mistake you can make
in RL is to underestimate the price of complexity.

Enhance your Environment. Add some features that make the problem a little more interesting.
Retrain with different versions and see how your changes alter the agent's learning. Aim for
something on the level of pong or snake. If you get something visually interesting and not already
in PufferLib, come show us and PR it! Many simple environments end up being quite useful in
research.

## 4. Start Understanding why that Worked

Read the core algorithm paper. It is dense, but the actual algorithm is simple. Read the bullet
point on PPO from my quickstart guide first for some intuition. Then read the Proximal Policy
Optimization (PPO) paper. You can ignore the TRPO and KL-penalty sections. The main equation 7 is
just saying: clip the policy gradient and weight it by the advantage function, then average over a
batch of data. The advantage function is Generalized Advantage Estimation (GAE). If you have a
strong math background, read this now. Otherwise, put it off until section 6.

Understand why I didn't have you implement the algorithm. The common reference algorithms are
extremely fiddly. Read Costa's 37 Implementation Details of PPO and the associated CleanRL PPO
implementation. It's not particularly long, but the details really matter.

Read our other articles. They are simpler than academic papers and tell you exactly how the tools
you have been using work.
- Stronger Hyperparameters with PROTEIN
- Puffing up PPO
- Neural MMO 3.0

## 5. Your First Real Project

Finish reading my quickstart guide. You will not have context for all of it just yet, but it
contains a lot of useful perspective as you start to solve harder problems. For example, depending
on how hard your environment is, you may actually have to run a hyperparameter sweep to get a good
policy. If you implemented something like a card game, you may at least need to decrease the
discount rate. Do you understand why?

Read the Pokemon Red RL Blog. This is a Powered by Puffer project that beats the entire game using
pure from-scratch reinforcement learning. The observation space formulation, reward engineering,
and problem setup are all informative. Note that Pokemon Red is ~1000x slower than most of our RL
environments. You will almost never need to actually do this level of engineering when you have
faster simulators.

Pick an interesting problem. Aim for something you can do in 500-1000 lines, depending on your
programming background. Several useful environments have been shorter than that. Arcade games are
usually a good choice. Just check in the Discord that nobody is already doing the same one. If you
have experience in another field, applied problems are even better. Good RL environments look like
fiddly interactive optimization problems that are quick to simulate and have clearly defined
observations and actions. The initial drone environment was only a few hundred lines.

Solve and PR to PufferLib. It may seem self-serving, but this is genuinely the best way to learn.
We have an active community of researchers and hobbyists large enough that someone will nearly
always be around to answer your questions. Many environments that you wouldn't expect to be useful
actually help us advance core RL research.

## 6. Read these Papers

The list of important papers to read in reinforcement learning is quite short. These are the top
10 papers you should read, regardless of what you want to do next. This list is intentionally not
a historical account of mostly broken algorithms. The focus is on the major capabilities-defining
results and the commonalities among them. Almost all of the OpenAI/DeepMind results have associated
blog posts that are more accessible than the formal manuscripts.

- **Dota 2 with Large Scale Deep Reinforcement Learning**: My pick for the most important paper in
  the field. PPO with a single-layer LSTM solves DoTA. Many top researchers undervalue this result
  and waste time developing fancy methods to solve trivial problems. Do not skip the appendix.
- **Grandmaster level in StarCraft II using multi-agent RL**: Another extremely hard problem solved
  with RL. #2 because DeepMind bootstrapped with imitation learning and used more complicated
  methods. They do this often, and I'm not convinced it is required.
- **Mastering the game of Go with deep neural networks and tree search**: Landmark historical result
  that kickstarted the field.
- **Learning Dexterous In-Hand Manipulation**: Rubik's cube with a robot hand. Pioneered domain
  randomization (training on a ton of slightly different problems) for generalization/robustness.
- **Open-Ended Learning Leads to Generally Capable Agents (XLand)**: Randomization over training
  tasks allows agents to generalize to new tasks not seen during training.
- **Emergent Tool Use From Multi-Agent Autocurricula**: 3v3 hide and seek with movable obstacles.
- **Human-level performance in first-person multiplayer games with population-based deep RL**: 3v3
  capture the flag with FPS mechanics.
- **The NetHack Learning Environment**: A really hard environment. You probably can't solve it with
  a general method without also solving AI. AI-complete.
- **Proximal Policy Optimization**: The core algorithm that is the basis of most modern RL.
  PufferLib 3.0's algorithm is a set of enhancements to PPO.
- **High-Dimensional Continuous Control Using Generalized Advantage Estimation**: The advantage
  function that is half the reason PPO works. PufferLib 3.0 combines GAE with VTrace.
- **Playing Atari with Deep Reinforcement Learning**: The original deep Q learning paper.

## 7. My Best Advice in One Place

**How to approach a new problem:** Start from first principles. The agent is learning tabula rasa -
it's a blank slate. At the start of training, it's looking for signal by mashing buttons. It also
can't see. Imagine the environment has the graphics completely randomized. Some reward has to be
obtainable this way. To actually learn from this reward, the agent needs to be able to see. What
information does the agent need to solve the problem? Make that the observation space. Ditto for
actions. What information would tell you if your agent is working correctly? Log that. **You almost
always want a single real number metric of overall performance. Don't only log raw reward, because
you will probably tune the scale of this number and make results incomparable. Log a score
instead.** For example, we might give the agent a reward of 0.25 or 0.5 for breaking a brick, but
we log the actual number of points obtained. Bonus points if you can scale it to the range of 0 to
1. Log any extra data you will actually look at, but don't log a ton of stuff you won't use. Good
candidates include collision rates, out of bounds, etc. since these are sanities that should drop
to near 0 for applicable problems. **Write the simplest possible environment that is still fast.
Don't abstract anything.** That's important enough to say twice: don't abstract anything. **Start
training early and frequently. You want iteration speed to be as fast as possible. Seconds is
better than minutes. You've lost if it is hours. If training does not work, suspect your data.**
Make your environment playable. Is what you see happening sensible? Do you see reward being
assigned when you expect it to be? **Run an evaluation on a checkpoint. See if the agents have
found some degenerate unrecoverable state.** For harder problems, scale up slowly and don't make a
ton of changes without training a decent model on the latest version. **Don't experiment with new
algorithms on new environments that are constantly changing. Do your research on stable problems,
then try it on the new environment.**

**How to encode data:** Normalize the observations appropriately. You want position data to be
egocentric when possible. You can do this by subtracting the agent's position and dividing by the
maximum value. Discrete data can't just go into model without encoding. If you represent knight,
king, queen, pawn, rook as 0, 1, 2, 3, 4, inputting that into a model raw implies knight is more
similar to king than it is to rook. It also forces the model to learn a wonky decision boundary in
latent space. One-hot encode knight as [1, 0, 0, 0, 0] instead. You can do this in the environment
for very small values, but do it in the policy for larger values to save bandwidth. Actions should
be the simplest possible set of controls for your environment. I like to imagine the environment
being released for gameboy and designing the controls to match. Neural MMO 3 features exploration,
combat, equipment, consumables, progression, and a live market. The action space is a single
discrete, and the game is playable with keyboard only. Don't blow up the size of the space.

## 8. Advanced Topic: Applications

RL works when you have a fiddly interactive optimization problem and can build a fast sim. PufferAI
does open-source work on drones and logistics (simulators that run millions of steps/second in a
few hundred lines of C). Longer term: hard science and manufacturing. Successful applications:
- Magnetic control of tokamak plasmas through deep RL (nuclear fusion)
- A Better Match for Drivers and Riders: RL at Lyft (rideshare logistics)
- Controlling Commercial Cooling Systems Using RL (datacenter cooling)
- Robust Autonomy Emerges from Self-Play (self-driving)
- Chip Placement with Deep RL (chip design)

## 9. Advanced Topic: Algorithms

More published results are wrong in RL than in other areas of AI. Even the ones with strong
evidence. Research areas with open questions:
- **Off-policy learning**: On-policy methods work great and are fast with unlimited data. Off-policy
  sometimes favored in data-poor settings. Papers: Rainbow; Beyond The Rainbow; Human-level Atari
  200x faster; IMPALA (introduces VTrace; PufferLib 3.0 tried it, worse than GAE alone, helped a
  little combined with GAE).
- **Model-based Learning**: Recurrent World Models; DreamerV3; "Reward Scale Robustness for PPO via
  DreamerV3 Tricks" (our publication showing the DreamerV3 tricks don't work by the given reasoning).
- **Search**: MuZero; Go-Explore (settable sim for hard exploration); EfficientZero V2.

## 10. Advanced Topic: Infrastructure

PufferLib models train at 3-5 million steps/second in PyTorch eager mode.
- **Environment speed**: Read `env_binding.h`. It defines `vec_reset`/`vec_step` - just N copies of
  your env run in a loop. The logging method iterates the fields of your `Log` struct and divides by
  `log->n`. Data is not copied between C and Python during step; both have access to the same
  pointers. Lowest-overhead possible. Next optimization: caching and SIMD.
- **Parallelization**: Python implementation of EnvPool. As of 3.0, round-robin N-way buffered by
  default, all buffers in shared memory. Each env writes directly to its memory block; the whole
  batch is available to the main thread without extra copies.
- **Models and Training**: The biggest opportunity. Some ops (entity embedding) lack custom kernels.
  Performance depends strongly on thousands of parallel environments and large minibatch sizes.
  Goal: less degradation with fewer environments and smaller minibatches.

---

### Project takeaways (how this guide maps to rotmg-rl)

- **Log a score, not raw reward** -> use boss-kill fraction / per-episode clear rate as the score
  (we're normalizing rewards separately, exactly as the guide warns).
- **Suspect your data when training fails** -> the reward-scale fix (rewards to -1..1) is step one.
- **Run an eval on a checkpoint; look for a degenerate state** -> watch the follow_along rollout of
  a trained checkpoint to SEE what the "training degrades the policy" behavior actually is.
- **Iteration speed: seconds, not minutes** -> the C env (3.39M SPS) is exactly this.
- **Scale up slowly; don't tune algos on a changing env** -> settle the env + reward, THEN sweep.
- **Egocentric, normalized observations** -> already done (local 31x31, presence/fraction channels).
