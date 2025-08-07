# Estudos sobre machine learning

Este repositório contém snippets e rascunhos de código que foram reunidos durante meus estudos sobre machine learning. Todo o material é feito em Python e está organizado de acordo com a biblioteca principal utilizada.

### PyTorch

Tutorial: [PyTorch for Deep Learning & Machine Learning – Full Course](https://www.youtube.com/watch?v=V_xro1bcAuA)

**Chapter 0 – PyTorch Fundamentals**

1. `28/07/2025` Why use machine/deep learning?
2. `28/07/2025` The number one rule of ML
3. `28/07/2025` Machine learning vs deep learning
4. `28/07/2025` Anatomy of neural networks
5. `28/07/2025` Different learning paradigms
6. `28/07/2025` What can deep learning be used for?
7. `28/07/2025` What is/why PyTorch?
8. `28/07/2025` What are tensors?
9. `28/07/2025` Outline
10. `28/07/2025` How to (and how not to) approach this course
11. `28/07/2025` Important resources
12. `29/07/2025` Getting setup
13. `29/07/2025` Introduction to tensors
14. `29/07/2025` Creating random tensors in PyTorch
15. `29/07/2025` Creating tensors with zeros and ones in PyTorch
16. `29/07/2025` Creating tensors in a range
17. `29/07/2025` Tensor datatypes
18. `29/07/2025` Getting tensor attributes (information about tensors)
19. `30/07/2025` Manipulating tensors (tensor operations)
20. `30/07/2025` Matrix multiplication (part 1)
21. `30/07/2025` Matrix multiplication (part 2: the two main rules of matrix multiplication)
22. `30/07/2025` Matrix multiplication (part 3: dealing with tensor shape errors)
23. `02/08/2025` Finding the min, max, mean and sum of tensors (tensor aggregation)
24. `02/08/2025` Finding the positional min and max of tensors
25. `02/08/2025` Reshaping, viewing and stacking tensors
26. `02/08/2025` Squeezing, unsqueezing and permuting tensors
27. `02/08/2025` Selecting data from tensors (indexing)
28. `02/08/2025` PyTorch and NumPy
29. `02/08/2025` PyTorch reproducibility (taking the random out of random)
30. `02/08/2025` Different ways of accessing a GPU in PyTorch
31. `02/08/2025` Setting up device agnostic code and putting tensors on and off the GPU
32. PyTorch Fundamentals exercises & extra-curriculum

**Chapter 1 – PyTorch Workflow**

33. `03/08/2025` Introduction to PyTorch Workflow
34. `03/08/2025` Getting setup for the PyTorch Workflow module
35. `03/08/2025` Creating a simple dataset using linear regression
36. `03/08/2025` Splitting our data into training and test sets (possibly the most important concept in machine learning)
37. `03/08/2025` Building a function to visualize our data
38. `03/08/2025` Creating our first PyTorch model for linear regression
39. `03/08/2025` Breaking down what's happening in our PyTorch linear regression model
40. `03/08/2025` Discussing some of the most important PyTorch model building classes
41. `04/08/2025` Checking out the internals of our PyTorch model
42. `04/08/2025` Making predictions with our random model using torch.inference_mode()
43. `04/08/2025` Training a model with PyTorch (intuition building)
44. `04/08/2025` Setting up a loss function and optimizer with PyTorch
45. `05/08/2025` PyTorch training loop steps and intuition
46. `05/08/2025` Writing code for a PyTorch training loop
47. `05/08/2025` Reviewing the steps in a PyTorch trainig loop
48. `05/08/2025` Running our training loop epoch by epoch and seeing what happens
49. `05/08/2025` Writing testing loop code and discussing what's happening
50. `05/08/2025` Reviewing what happens in a testing loop step by step
51. `06/08/2025` Writing code to save a PyTorch model
52. `06/08/2025` Writing code to load a PyTorch model
53. `07/08/2025` Getting ready to practice everything we've done so far with device agnostic-code
54. `07/08/2025` Putting everything together part 1: preparing data
55. `07/08/2025` Putting everything together part 2: building a model
56. `07/08/2025` Putting everything together part 3: training a model
57. `07/08/2025` Putting everything together part 4: making predictions with a trained model
58. `07/08/2025` Putting everything together part 5: saving and loading a trained model
59. PyTorch Workflow exercises & extra-curriculum

**Chapter 2 – Neural Network Classification**

60. Introduction to machine learning classification
61. Classification input and outputs
62. Architecture of a classification neural network
63. 
64. Turing our data into tensors
65. 
66. Coding a neural network for classification data
67. 
68. Using torch.nn.Sequential
69. Loss, optimizer and evaluation functions for classification
70. From model logits to prediction probabilities to prediction labels
71. Train and test loops
72. 
73. Discussing options to improve a model
74. 
75. 
76. Creating a straight line dataset
77. 
78. Evaluating our model's predictions
79. The missing piece – non-linearity
80. 
81. 
82. 
83. 
84. Putting it all together with a multiclass problem
85. 
86. 
87. 
88. Troubleshooting a mutli-class model
89. 
90. 
91. 

**Chapter 3 – Computer Vision**

92. Introduction to computer vision
93. Computer vision input and outputs
94. What is a convolutional neural network?
95. TorchVision
96. Getting a computer vision dataset
97. 
98. Mini-batches
99. Creating DataLoaders
100. 
101. 
102. 
103. Training and testing loops for batched data
104. 
105. Running experiments on the GPU
106. Creating a model with non-linear functions
107. 
108. Creating a train/test loop
109. 
110. 
111. 
112. Convolutional neural networks (overview)
113. Coding a CNN
114. Breaking down nn.Conv2d/nn.MaxPool2d
115. 
116. 
117. 
118. Training our first CNN
119. 
120. Making predictions on random test samples
121. Plotting our best model predictions
122. 
123. Evaluating model predictions with a confusion matrix
124. 
125. 

**Chapter 4 – Custom Datasets**

126. Introduction to custom datasets
127. 
128. Downloading a custom dataset of pizza, steak and sushi images
129. Becoming one with the data
130. 
131. 
132. Turning images into tensors
133. 
134. 
135. 
136. Creating image DataLoaders
137. Creating a custom dataset class (overview)
138. 
139. Writing a custom dataset class from scratch
140. 
141. 
142. Turning custom datasets into DataLoaders
143. Data augmentation
144. Building a baseline model
145. 
146. 
147. Getting a summary of our model with torchinfo
148. Creating training and testing loop functions
149. 
150. 
151. Plotting model 0 loss curves
152. Overfitting and underfitting
153. 
154. 
155. Plotting model 1 loss curves
156. Plotting all the loss curves
157. Predicting on custom data

Links úteis:

* [Zero to Mastery Learn PyTorch for Deep Learning](https://www.learnpytorch.io/)
* [pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning)
