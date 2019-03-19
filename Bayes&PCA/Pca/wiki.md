# Machine Learning basis Wiki
## PCA-Classifier

### Intro
Idea: Given some dataPoints in a certain d-dimentional space, project them into a lower 
dimentional space while preserving as much information as possible.<br/>
Some examples:<br/>
Find the best planar approximation to a 3-D image  
Find the best 12-D approximation to a 10^(14) data

In particular to chose a projection that **maximize squared error** this is the factor that
permits us to catch the more info possible.

### Definition
Given a certain dataset, PCA is an algorithm that permits an orthogonal projection 
of the data onto a lower-dimentional linear space  
- that maximize data of the projected data;
- minimize min squared distance between data-point and projection

### Approches
There are 2 main approches for the PCA algo:
- Sequential (through PC vectors) 
- Through Covariance Matrix
- SVD of data matrix

#### PCA through PC-vectors (principal components vectors)
PCA permits the transportation of data onto a lower dimension by computing the k-PC vectors
of the selected dataset, and using only a set of them to reproject the image.<br/>
Properties:<br/>
- Those vectors originate from the center of mass. (in the example PC(2) produce exactly a point
in the center of mass we are aiming to);
- The first PC(#1) point on the direction of the largest variance;
- Any other is orthogonal of the first one point on the largest variance in the residual data-space
(the more we extract the less information we get);

#### PCA through Covariance Matrix
Given a Dataset {x1...xn}, it consist on calculate the sample Covariance Matrix by subtracting 
for each row xi the mean value (**standardization**). [X] <br/> 
Then the **covariance matrix** would be equal to Cov = X * X^ (where x is the transposition of X)<br/>

Then the remaining task will be to obtain from this matrix the **eighenvalue** and **eighenvectors**.
<br/>{&#955;i, ui}
   
The eighenvector of the cov-Matrix are the basic component of the PCA, and the more the 
associated eighenvalue is larger the more important is the associated eighenvector.<br/>
I mean, the more the eighenvalue, the more the eighenvector maximize the variance (so the more
would be the info that we are gathering about the model)

So, after obtain them we have to sort them in Descending order and take the set base onto the 
number of components we want to take in consideration to reduce the data-space (**top-k eighenvectors**).
<br/>{&#955;1> &#955;2>... >&#955;n}
<br/>(take the top(first) k=50 of them)
<br/>

##### This is also the technique adopted in the sketch for thecalculation of the last 6 PCs

## PCA from Scratch scratch
This is an extract from the code that i had post.
```python
import numpy as np
def pca_from_scratch(n_components, first_or_last, std_matrix):
    # Obtain the Eigenvectors and Eigenvalues from the covariance matrix or
    # correlation matrix, or perform Singular Vector Decomposition.
    cov_X = np.cov(std_matrix)
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_X)

    # Raises an AssertionError if two objects are not equal up to desired precision.
    # The test verifies identical shapes and that the elements of actual and desired satisfy.
    for ev in eig_vec_cov:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:, i]) for i in range(len(eig_val_cov))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    # reverse=False == Ascending, so from the one with highest variance to minimum
    if first_or_last is True:
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
    else:
        eig_pairs.sort(key=lambda x: x[0], reverse=False)

    matrix_w = np.hstack([eig_pairs[i][1].reshape(1087, 1) for i in range(n_components)])
    return matrix_w
```

This code is exactly the 2nd Algorithm that I explaned before, the one with the sample covariance 
matrix to be clear.<br/>
It could be devided in 5 steps:
- standardize matrice X
- compute covariance matrix
- extract eighenvalue and eighenvector from covariance matrix
- sort tuples (eighenvalue, eighenvector) by the eighenvalues
(ascendent)
- compute matrix with first k PCs

####Standardize matrice of samples X
We can avoid this passage because we just have as input that one. However the function to use
to standardize it is:
```python
# standardization of feature matrix
# makes feature_matrix with mean 0 & variance 1 -> as definition od standardization
std_X = (X - np.mean(X)) / np.std(X)
```
Just for clarification, standardization is the process of putting different variables on the 
same scale.</br>
Otherwise, variables measured at different scales do not contribute equally to the analysis. 
<br/>For example, in boundary detection, a variable that ranges between 0 and 100 will 
outweigh a variable that ranges between 0 and 1. Using these variables without standardization 
in effect gives the variable with the larger range a weight of 100 in the analysis.

####Compute covariance matrix
covariance matrix (also known as dispersion matrix or variance–covariance matrix) is 
a matrix whose element in the i, j position is the covariance between the i-th and j-th 
elements of a random vector.<br/>
Intuitively, the covariance matrix generalizes the notion of variance to multiple dimensions. 
<br/><br/>As an example, the variation in a collection of random points in two-dimensional 
space cannot be characterized fully by a single number, nor would the variances in the 
x and y directions contain all of the necessary information; a 2×2 matrix would be necessary 
to fully characterize the two-dimensional variation. 

for this operation I used a bases instruction of the numpy lib:
```python
cov_X = np.cov(std_matrix)
```

####Extract eighenvalue and eighenvector from covariance matrix
for extract the eighenvalue and eighenvector from the covariance matrix I used the instruction:
```python
eig_val_cov, eig_vec_cov = np.linalg.eig(cov_X)
```
The eigenvectors represent the directions or components for the reduced subspace of B, whereas 
the eigenvalues represent the magnitudes for the directions.

The eigenvectors can be sorted by the eigenvalues in descending order to provide a ranking 
of the components or axes of the new subspace.
From this point the next point

#### Sort tuples (eighenvalue, eighenvector) by the eighenvalues
As we said before at this point what we have to do is to 
 >"sort them in Descending order"
 
This will give us a rank starting from the top principal component for reprojecting the image.<br/>

About that, note that we can also choose to sort them in Ascending order 
as we did in this piece of code:<br/>
```python
# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:, i]) for i in range(len(eig_val_cov))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
# reverse=False == Ascending, so from the one with highest variance to minimum
if first_or_last is True:
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
else:
    eig_pairs.sort(key=lambda x: x[0], reverse=False)
```
That's non a mistake, its only for show you how much it change to take the last 6 instead 
of the first 6.<br/>

In this extract at first we combine every eighenvector to his own eighenvalue in a list of tuple,
then we sort this one based on a parameter in input.<br/>
As, it's written in the comment below we have a Descending(normal) order in case we pass False
as parameter, Ascending otherwise.<br/>

#### Compute matrix with first k PCs
A total of m or less components must be selected to comprise the chosen subspace. Ideally, we 
would select k eigenvectors, called principal components, that have the k largest eigenvalues.

```python
matrix_w = np.hstack([eig_pairs[i][1].reshape(1087, 1) for i in range(n_components)])
```
This istruction extracts from every touple the corresponding eighenvector, transpose it in 
(1087*1).<br/> 
All those eighenvector are just listed as [ev[1]...ev[n]] at the end of this phase, so we have
only to stack those in sequence horizontally (column wise) to end this phase.<br/>

After this phase we have the matrix_w ready to be plotted. So, we got our data in a reducted
dataspace, ready to be used for our purpose.

## Uses and problem
The algorithm that we have summarize is very usefull for a bunch of stuff, like:

- Get a compact description of the data, summarize them decreasing the dimentionality
- Ignore noise, infact taking the k-PCs decrese the noise inserted in the image
- Improve classification (Hopefully)

At contrary it can also create some kind of problem. One of them can be easily seen in the example.
<br/>
For example if we take the whole dataset, that contains image of various class toghether,
what we risk in applying the PCA is that:<br/>
If there are lot more image of one class with the respect to the other, could happen that 
the main k-PCs will be the one that mainly characterize that class.
It means that for example, our image that is from another class will be mainly recostructed with
PCs suited for other class... so the projection will be very different respect to the real one.
<br/>
Other limitation of PCA are:
- Relies on linear asumptions: <br/>
PCA is focused on finding orthogonal projections of the dataset that contains the highest 
variance possible in order to 'find hidden LINEAR correlations' between variables of the dataset.
This means that if you have some of the variables in your dataset that are linearly correlated, 
PCA can find directions that represents your data.<br/>
But if the data is not linearly correlated (f.e. in spiral, where x=t*cos(t) and y =t*sin(t) ), 
PCA is not enough

- Relies on orthogonal tranformations: <br/>
Sometimes consider that principal components are orthogonal to the others it's a restriction 
to find projections with the highest variance
- Scale variant: <br/>
PCA, as you could've seen, is a rotation trasnformation of your dataset, wich means that doens't
affect the scale of your data. It's worth to said also that in PCA you dont normalize your data.
That means that if you change the scale of just some of the variables in your data set, you 
will get different results by applying PCA.


