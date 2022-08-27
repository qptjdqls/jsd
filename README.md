# jsd

## Original code: <br>
### https://stats.stackexchange.com/questions/345915/trying-to-implement-the-jensen-shannon-divergence-for-multivariate-gaussians/419421#419421 <br>

Compare Jensen-Shannon divergence of <br>
(1) joint distribution and the product of mariginal distribution $JSD(p(x _1, x _2, ..., x _n) \| \| p(x _1)p(x _2)\dots p(x _n))$ with <br>
(2) other distributions $JSD(p(x _1, \dots, x _k) \| \| p(x _1, \dots, x _{k-1})p(x _k))$ ($k=1,\dots, n$) <br>