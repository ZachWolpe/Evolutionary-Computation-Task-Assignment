# Evolutionary Computation

see pdf for report

Software Implementation of Metaheuristics:

### Genetic Algorithms, Simulated Annealing & Particle Swarm Optimization 

A first principles implemenation of GA, SA & PSO to solve a 2^150 knapsack instance.


# Knapsack Problem

Knapsack is a combinatorial search space problem defined as follows:
Given a list of items each containing a ‘value’ & a ‘weight’
We can only take a certain amount of items, limited by the total weight of the items selected; thus we have a maximum Capacity that cannot be exceeded (the maximum what of fulling a knapsack).
We wish to take a combination of items that Maximizes the Total Value of the selected items (items in the knapsack).


Whilst seemingly simplistic, the knapsack problem becomes very difficult as the number of items balloons: as the number of combinations grows exponentially as a function of the number of items.  

The given Knapsack contains 150 items. Each item can either be 
	- Selected: 1 
	- Not Selected: 0 
Resulting in:     
   
   2^150 = 1.4272477e+45
				        
Permutations.



Consequently randomly search (trying all possibly combinations) is infeasible & would take millions of years to compute. We thus rely on intelligent search algorithms to attempt many possible combinations in a sophisticated way to find an elegant solution. 
Though Knapsack is a toy problem, one can easily imagine how this algorithm could be applied to ANY combinatoric search space (& in fact is generalizable to any complex search domain).







# Specifications

The given knapsack:
Combinatorial space: 	       	2^150 = 1427247692705960000000000000000000000000000000
Maximum capacity: 		822
Maximum iterations:		10’000
Best known optimum:	997

The same problem is solved by 3 algorithms: Genetic Algorithms (GA), Simulated Annealing (SA) & Particle Swarm Optimization (PSO). Full algorithm descriptions are available on the GitHub repository.



------------------------------------------------------------------------------------------------------------------------------


# Results

Each algorithm was run for a varied spectrum of hyper parameters (defining changes in basic behaviour like the cooling rate of SA or the number of particles in a swarm in PSO). Performance of each specification was measured by ‘squality’ or ‘Solution Quality’ P which compares the best learnt solution for a given algorithm (maximum value of the knapsack) with best known solution of 997 - computed by random search on a super-computer. Squality gives the best performance as percentage of 997.

Given the limited computational power, an squality >= 0.8 would be considered successful, however all algorithms exceeded this benchmark, with GA & PSO maxing out at squality > 1 (meaning better than the best known solution). 


These truly remarkable algorithms are capable of effective search in inconceivably sparse high dimensional spaces. Note, whilst these applications are applied to discrete problem the same techniques are readily applied to continuous search spaces with minor re-specifications (simple genotype encoding, phenotype encoding & fitness evaluation updates). Thus applications may be extended to any domain that can be mapped to a theoretical mathematical function.




 
