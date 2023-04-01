import numpy as np

def mutual_information(A, B, L=256):
    # MI Determines the mutual information of two images or signals
    #
    #   I=mi(A,B)   Mutual information of A and B, using 256 bins for
    #   histograms
    #   I=mi(A,B,L) Mutual information of A and B, using L bins for histograms
    #
    #   Assumption: 0*log(0)=0
    #
    #   See also ENTROPY.
    #
    #   jfd, 15-11-2006
    #        01-09-2009, added case of non-double images
    #        24-08-2011, speed improvements by Andrew Hill

    A = np.double(A)
    B = np.double(B)

    # LB is assumed to be RESP, so it will have fewer bins (LB is ~10)
    LB = min(L, len(np.unique(B)))
    nb, _ = np.histogram(B.flatten(), bins=LB)
    nb = nb / np.sum(nb)

    # LA = min(L, length(unique(A)));
    LA = L  # 10 was too low, 15 is ok, and 20-25 is pretty stable.
    na, _ = np.histogram(A.flatten(), bins=LA)
    na = na / np.sum(na)

    n2 = hist2(A, B, LA, LB)
    n2 = n2 / np.sum(n2)  # P(A,B)

    papb = np.outer(na, nb)  # P(A)P(B)
    I = np.sum(minf(n2, papb))

    return I, n2


def minf(pab, papb):
    I = np.logical_and(papb > 1e-12, pab > 1e-12)  # Pick only values > 10^-15
    y = pab[I] * np.log2(pab[I] / papb[I])  # Defn of Mutual Information
    return y


def hist2(A, B, LA, LB):
    """
    Calculates the joint histogram of two images or signals.

    n = hist2(A, B, L) is the joint histogram of matrices A and B, using L bins for each matrix.
    """

    # Compute minimum and maximum values of A and B
    ma = np.min(A)
    MA = np.max(A)
    mb = np.min(B)
    MB = np.max(B)

    # For sensorimotor variables, in [-pi,pi]
    # ma = -np.pi
    # MA = np.pi
    # mb = -np.pi
    # MB = np.pi

    # Scale and round to fit in {0,...,L-1}
    A = np.round((A - ma) * (LA - 1) / (MA - ma + np.finfo(float).eps))
    B = np.round((B - mb) * (LB - 1) / (MB - mb + np.finfo(float).eps))
    n = np.zeros((LA, LB))
    x = np.arange(LB+1)

    for i in range(LA):
        if np.sum(A==i):
            n[i, :] = np.histogram(B[A == i], bins=x, density=True)[0]

    return n