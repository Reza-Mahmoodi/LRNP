## Link prediction and its optimization based on low-rank representation of network structures (2023)
- Authors = Chai, Lang and Tu, Lilan and Yu, Xinyi and Wang, Xianjia and Chen, Juan
- Journal = Expert Systems with Applications
### Abstract
Currently, among the link prediction researches based on low-rank representation, almost none of literature has considered how to select an appropriate base-matrix and the impact of the structural characteristics of the reconstructed network on the link prediction. Therefore, in this paper, we use the adjacency matrix of the fully connected network (FCN) as the base-matrix for low-rank representation, and any local structure of the observed networks can be represented by the interactions of FCN structures. To explore the properties of link predictions, the nuclear norm of the adjacency matrix for the reconstructed network is taken as a penalty term in the newly proposed low-rank representation objective function. According to the optimal interactive coefficients achieved by solving the novel objective function, in this paper, we design a novel link prediction algorithm (LRNP algorithm) and its optimized algorithm (OLRNP algorithm). Experimental results based on real networks and synthetic networks lead to several conclusions. (1) The LRNP algorithm has good convergence properties. When changing the parameters of the LRNP algorithm, the changes along prediction performance do not exceed 9.48%. LRNP also performs well for sparse networks. (2) Compared with baseline link prediction algorithms, LRNP also shows excellent performance, and its AUC and Precision can increase by 14.35% and 14.89%. (3) The OLRNP algorithm exhibits better performance than the LRNP algorithm, and its AUC and Precision can rise by up to 7.50% and 6.79%, respectively.

If you find this implementation helpful in your work, please consider citing both the original paper and our related research on **Link prediction by adversarial NMF**:

## Original Paper:

```
@article{chai2023link,
  title={Link prediction and its optimization based on low-rank representation of network structures},
  author={Chai, Lang and Tu, Lilan and Yu, Xinyi and Wang, Xianjia and Chen, Juan},
  journal={Expert Systems with Applications},
  volume={219},
  pages={119680},
  year={2023},
  publisher={Elsevier}
}
```

## Our Paper:
```
@article{mahmoodi2023link,
  title={Link prediction by adversarial nonnegative matrix factorization},
  author={Mahmoodi, Reza and Seyedi, Seyed Amjad and Tab, Fardin Akhlaghian and Abdollahpouri, Alireza},
  journal={Knowledge-Based Systems},
  volume={280},
  pages={110998},
  year={2023},
  publisher={Elsevier}
}
```
