alpha=0 # hidden nodes
beta=1  # free parameter
x=2     # irt

# math.gamma or math.lgamma?
((beta**alpha)/math.lgamma(alpha))*x**(alpha-1)*math.e**(-beta*x)

