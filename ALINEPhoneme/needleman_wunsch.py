import numpy as np

def needleman_wunsch(seq1, seq2, gap_penalty=-1, match_score=1, mismatch_penalty=-1):
    n, m = len(seq1), len(seq2)
    dp = np.zeros((n + 1, m + 1), dtype=int)

    for i in range(n + 1):
        dp[i][0] = gap_penalty * i
    for j in range(m + 1):
        dp[0][j] = gap_penalty * j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = dp[i-1][j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch_penalty)
            delete = dp[i-1][j] + gap_penalty
            insert = dp[i][j-1] + gap_penalty
            dp[i][j] = max(match, delete, insert)

    align1, align2 = [], []
    i, j = n, m
    while i > 0 and j > 0:
        if dp[i][j] == dp[i-1][j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch_penalty):
            align1.insert(0, seq1[i-1])
            align2.insert(0, seq2[j-1])
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i-1][j] + gap_penalty:
            align1.insert(0, seq1[i-1])
            align2.insert(0, '-')
            i -= 1
        else:
            align1.insert(0, '-')
            align2.insert(0, seq2[j-1])
            j -= 1

    while i > 0:
        align1.insert(0, seq1[i-1])
        align2.insert(0, '-')
        i -= 1
    while j > 0:
        align1.insert(0, '-')
        align2.insert(0, seq2[j-1])
        j -= 1

    return align1, align2