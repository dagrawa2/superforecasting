This repository contains an implementation of what I call a variance-loss neural network (VLNN), which I developed while working on a project on superforecasting with MIT Lincoln Lab.

A VLNN is similar to a Bayesian network of binary random variables. But instead of weighting every node A with in-neighbors B and C by a multi-dimensional conditional probability table P(A=i | B=j, C=k), we weight the edges from B to A and C to A by the square contingency tables P(A=i | B=j) and P(A=i | C=k) respectively. A VLNN is initialized with "guestimates" for these conditional probability edge weights as well as for the probabilities of nodes with no in-neighbors. Then, the parameters of the VLNN are adjusted so as to satisfy the law
	P(A) = P(A | B)*P(B) = P(A | C)*P(C).
That is, we are minimizing the variance in the values of P(A) as computed along every incoming edge incident to A. The result as a network of statistical dependencies in which the conditional probabilities are mathematically valid.

A VLNN is therefore used to correct human guestimates for conditional probabilities.

This repository includes both a python script as well as a PDF document detailing the implemented equations.