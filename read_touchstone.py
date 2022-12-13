import skrf as rf

network = rf.Network('data/trans_short.s2p')
network.plot_s_smith(lw=2)
print(network.f)