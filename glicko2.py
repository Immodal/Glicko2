# Written by Nigel Chin based on the Glicko2 system by Professor Mark E Glickman
# Source: http://www.glicko.net/glicko/glicko2.pdf

import numpy as np

SCALER = 173.7178
BASE = 1500
TAU = 0.5


def predict(r_a, r_b, rd_b):
    '''
    Takes the ratings for players A and B, and Rating Deviation of B.
    Returns the probability of A winning.
    '''
    return E(to_mu(r_a), to_mu(r_b), to_phi(rd_b))


def update(r_a, rd_a, vol_a, rs_b=None, rds_b=None, vols_b=None, scores=None):
    '''
    Given the ratings, ratings deviation and volatility of player A and B, as well as the score,
    return a tuple of the updated values for player A.

    r_a, r_d, vol_a should be single numeric values.

    rs_b, rds_b, vols_b should be numpy arrays containing the values for all opponents in defined a period,
    but they can also be single numeric values.
    '''
    if rs_b is not None:
        # Step 2
        mu_a = to_mu(r_a)
        phi_a = to_phi(rd_a)
        mus_b = to_mu(rs_b)
        phis_b = to_phi(rds_b)
        
        # Step 3 to 5
        var_val = variance(mu_a, mus_b, phis_b)
        d_val = delta(mu_a, mus_b, phis_b, scores, var_val)
        vol_new = volatility(mu_a, phi_a, vol_a, var_val, d_val)

        # Step 6. Update the rating deviation to the new 
        # pre-rating period value
        phi_pre = np.sqrt(phi_a**2+vol_new**2)

        if var_val!=0:
            # Step 7. Update the rating and RD to the new values
            phi_new = 1/np.sqrt(1/(phi_pre**2)+1/var_val)
            # the sum portion of the mu_new formula is the same as the 
            # delta calculation without v
            mu_new = mu_a + (phi_new**2)*(d_val/var_val)
        else:
            phi_new = 1/np.sqrt(1/(phi_pre**2))
            mu_new = mu_a

        return to_r(mu_new), to_rd(phi_new), vol_new
    else:
        # If no opponents, it is assumed that the player was inactive
        phi_a = to_phi(rd_a)
        phi_new = np.sqrt(phi_a**2+vol_a**2)

        return r_a, to_rd(phi_new), vol_a


def to_mu(r):
    return (r-BASE)/SCALER


def to_r(mu):
    return SCALER*mu+BASE


def to_phi(rd):
    return rd/SCALER


def to_rd(phi):
    return phi*SCALER

    
def variance(mu_a, mus_b, phis_b):
    '''
    Step 3. Calculate the estimated variance of the player's
    rating based only on game outcomes.
    '''

    g_vals = g(phis_b)
    E_vals = E(mu_a, mus_b, phis_b)
    total = np.sum((g_vals**2)*E_vals*(1-E_vals))

    return 1/total if total!=0 else 0

    
def g(phi):
    return 1/(np.sqrt(1+3*(phi**2)/(np.pi**2)))


def E(mu_a, mus_b, phis_b):
    '''
    This can be used to extract a win probability when predicting an outcome
    '''

    return 1/(1+np.exp(-g(phis_b)*(mu_a-mus_b)))


def delta(mu_a, mus_b, phis_b, scores, var_val):
    '''
    Step 4. Compute the quantity delta, the estimated improvement
    in rating by comparing thepre-period rating to the performance 
    rating based only on game outcomes. 
    
    scores are the scores against each opponent. 
    (0 for a loss,0.5 for a draw, and 1 for a win)
    '''

    g_vals = g(phis_b)
    E_vals = E(mu_a, mus_b, phis_b)
    total = np.sum(g_vals*(scores-E_vals))
    
    return var_val*total


def volatility(mu, phi, vol, var_val, d_val):
    '''
    Step 5. Determine the new value of the volatility. 
    '''
    def f(x):
        # Step 5.1
        ex = np.e**x
        bracket1 = d_val**2-phi**2-var_val-ex
        bracket2 = phi**2+var_val+ex
        bracket3 = x-a

        return ex*bracket1/(2*bracket2)-bracket3/TAU**2

    def get_B():
        if (d_val**2) > (phi**2+var_val):
            return np.log(d_val**2-phi**2-var_val)
        else:
            k = 1
            while True:
                if f(a-k)<0: k+=1
                else: return a - k*TAU
                
    def step(A, fA, B, fB):
        C = A + (A-B)*fA/(fB-fA)
        fC = f(C)
        if fC*fB<0:
            A = B
            fA = fB
        else:
            fA/=2
        B = C
        fB = fC
        
        return A, fA, B, fB
    
    a = np.log(vol**2)
    # Convergence tolerance
    eps = 0.000001
    
    # Step 5.2, Set initial values of iterative algorithm
    A = a
    B = get_B()
    
    # Step 5.3
    fA = f(A)
    fB = f(B)
    
    # Step 5.4, Iterate
    while abs(B-A)>eps:
        A, fA, B, fB = step(A, fA, B, fB)
    
    return np.e**(A/2)


def test():
    op_rs = np.array([1400,1550,1700])
    op_rds = np.array([30,100,300])
    op_vols = np.array([0.06,0.06,0.06])
    scores = np.array([1,0,0])

    r, rd, vol = update(1500, 200, 0.06, op_rs, op_rds, op_vols, scores)
    print("Test")
    print(f"{r:.2f} == 1464.06")
    print(f"{rd:.2f} == 151.52")
    print(f"{vol:.5f} == 0.05999")
    
    for i in range(5):
        r, rd, vol = update(1500, 200, 0.06, np.array([]), np.array([]), np.array([]),np.array([]))
    print(f"{r:.2f} == 1500")
    print(f"{rd:.2f} == 200.27")
    print(f"{vol:.5f} == 0.05999")

    r1, rd1, vol1 = update(1500, 200, 0.06, 1500, 200, 0.06, 1)
    r2, rd2, vol2 = update(1500, 200, 0.06, 1500, 200, 0.06, 0)
    print("Test2")
    print(f"r1 = {r1:.2f}")
    print(f"r2 = {r2:.2f}")
    print(f"{(r1 - 1500) + (r2 - 1500)} == 0")
