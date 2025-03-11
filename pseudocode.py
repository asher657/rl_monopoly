"""
# first episode: agent has 150$, opponent position = 0
# for steps until agent or opponent bankrupt

# do we want to remove houses per step
#       if the goal of the agent is to just determine the optimal place to buy house based on
#       opponent position, current money
#       if agent loses all money because opponent didn't land on house, then loss
#
# do we want win/loss ratio (

# what if we place houses on all properties (have we already lost at this point)
# What is the


### FINAL ###
assume we don't remove house
what is terminal state: when the agent or opponent is bankrupt
go gives opponent money
agent forced to buy a single property every step
if agent tries to buy house on property with house, lose money but don't increase rent
reward: the rent received if opponent lands on house, else cost to place a house
state space: 22 buyable properties, 40 opponent positions, 22 property rents, agent money (continuous), opponent money (continuous)
action space: 22 properties

                        Actions
(buyable, position) | [all the properties can buy, size 22]
"""