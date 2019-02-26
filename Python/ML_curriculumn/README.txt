NAME OF THE PROJECT: 
TEAM MUMBER: Yi Wang, Lina Cao, Jiaheng Zhou 

#1 word file ( final report) 
#1 READ ME TEXT FILE 
#2 CVS FILE FOR TEST AND TRAIN DATA SET ( WE TRANSFORM OUR DATA SET INTO CSV VERSION FOR EASY LOADING) 
#1 PYTHON FILE ( INCLUDE ALL THE PYTHON CODE USED FOR THE PROJECT) 

 
There are 3 of self-defined function:  (please add the defined function you created) 
1,learnWbL(X,y,Xtest,ytest,epsi,lam): 
 learn w, considering both accuracy and objective function as quit criteria;
 take epsilon and lambda as input variable for future decision.
 # INPUT
 X,y: explanatory variable matrix and response variable(0/1) in training set
 Xtest,ytest: explanatory variable matrix and response variable(0/1) in testing set (the one split from training set)
 epsi, lam: two hyper parameters, epsilon and lambda
 # OUTPUT
 k: iteration round
 accuracy: the correct rate when we get the updated w
 t: time that the function runs
 w: the updated weight

2,smartchoose(epsilist,lamlist):
 given two list of number for hyper parameters and select the combination that has a greater accuracy, return the combination group number to track the corresponding epsilon and lambda.
 # INPUT
epsilist,lamlist: the list of number that is to be chosen as epsilon and lambda 
 # OUTPUT
 tracki: under which combination that the accuracy is the highest
 
3,corrate(Xtest,ytest,w):
 # INPUT
 Xtest: the real test set of X
 ytest: the real test set of y
 w: the "optimal" w we get from learning
 # OUTPUT
 accuracy in test set