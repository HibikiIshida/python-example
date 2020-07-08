from scipy.stats import norm

alp = 0.01

za, zb = norm.interval(alpha=(1-alp), loc=0, scale=1)

print('Za = ', za)
print('Zb =  ', zb)