# Polynomial Regression with PyTorch
Explore a PyTorch-based Polynomial Regression implementation, featuring data visualization, synthetic dataset creation, and noise management. It exemplifies a simple ML pipeline, demonstrating proficiency in PyTorch, Matplotlib, Seaborn, and Numpy.

This project involves the development and evaluation of a Polynomial Regression model, implemented in PyTorch, on synthetic datasets. The aim is to estimate the parameters of a specific polynomial function given by 
$$p(z) = 0.05z^4 + z^3 + 2z^2 - 5z = \sum _{i = 0}^4 \textbf{w}_iz^i$$
which has been expressed as a dot-product between two vectors: $\textbf{w} = [0, −5, 2, 1, 0.05]^T$, and $\textbf{[1, z, z2, z3, z4]}$, so $p(z) = \textbf{w}^T\textbf{x}$.

We define a synthetic dataset $D$, where each instance is an $(x, y)$ pair, and y is calculated from the polynomial function with an added noise $\epsilon$. The noise follows a normal distribution with zero mean and standard deviation of $0.5$.

As part of the project, we will generate visualizations to gain an intuitive understanding of the polynomial function and its properties. Furthermore, we will investigate the performance of our model with different configurations and understand how the noise affects the model's ability to estimate the parameters.

This report presents the process, results, and learnings from the project in a detailed and systematic manner.

![ Output plot from the function provided in the assignment instructions. The polynomial plotted in blue corresponds to a $p(x) = \sum w_i x^i$ with the coefficients $\textbf{w} = [0,−5,2,1,0.05]$. Only values spanning from $x = −3$ up to $x = 3$ are shown](images/Polynomial_plot.png)
|:--:|
| <b>Figure 1:</b> Output plot from the function provided in the assignment instructions. The polynomial plotted in blue corresponds to a $p(x) = \sum w_i x^i$ with the coefficients $\textbf{w} = [0,−5,2,1,0.05]$. Only values spanning from $x = −3$ up to $x = 3$ are shown|
