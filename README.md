# Improving Personalized project recommendation on GitHub based on deep matrix factorization
This personalized recommendation system based on deep matrix factorization applying on GitHub. With the use of deep neural network, we learn a low dimensional representation of users and projects based on user-project matrix in a common space, in which we can capture the user's latent behavior preferences of each item, and automatically recommend to users the top N results.
This is our official implementation for the paper:
<!-- Xiangnan He, Zhankui He, Xiaoyu Du & Tat-Seng Chua. 2018. **Adversarial Personalized Ranking for Recommendation**  , In *Proceedings of SIGIR'18*.   
(Corresponding Author: [Dr. Xiangnan He](http://www.comp.nus.edu.sg/~xiangnan/)) -->
If you use the codes, please cite our paper . Thanks!
## Environment
- Python 3.7
- TensorFlow >= r1.13

PS. For your reference, our server environment is Intel(R) Core(TM) i5-9400 CPU @ 2.90GHz and 8 GiB memory. We recommend your free memory is more than 8 GiB to reproduce our experiments.

## Dataset
we collect user-project data from three organizations and the large-scale Google repository on GitHub. We'll introduce the details of our dataset with the example of Formidable.
| Group name | Users|	Projects |	Development areas	|Density	|Sparseness|
|---|---|---|---|---|---|
|vim-jp 	|47	|6262|	Vimscript	|9906	|0.03%|
|Formidable |	47	|2321	|Web 	|3067	|0.03%|
|harvesthq 	|31	|944	|Android	|1116|	0.04%|
|Large(Google)|	2663	|75417	|--	|123577	|0.06%|

**data\FormidableLabs\FormidableLabs_matrix_c_f_s.csv:**
- original data
- This file contains the behavior scores of all users in the organization for the project：userID\ projectID\ create or fork rating\ timestamp\ star rating\ timestamp (if have)

**data\FormidableLabs\translated_train1.csv:**
- Train file.
- Each Line is a training instance: userID\ projectID\ rating\ timestamp (if have)

**data\FormidableLabs\translated_test1.csv:**
- Test file (positive instances).
- Each Line is a testing instance: userID\ projectID\ rating\ timestamp (if have)
