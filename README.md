# Glicko2

Vectorized using numpy to handle large simultaneous updates.

```
op_rs = np.array([1400,1550,1700])
op_rds = np.array([30,100,300])
op_vols = np.array([0.06,0.06,0.06])
scores = np.array([1,0,0])

old_r = 1500
old_rd = 200
old_vol = 0.06

new_r, new_rd, new_vol = update(old_r, old_rd, old_vol, op_rs, op_rds, op_vols, scores)
```
