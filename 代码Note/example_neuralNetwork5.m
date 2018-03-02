%用作为卷积网络的主函数
tic
load digits.mat
[n,d] = size(X);
nLabels = max(y);
yExpanded = linearInd2Binary(y,nLabels);
t = size(Xvalid,1);
t2 = size(Xtest,1);

%input weights

% Standardize columns and add bias
[X,mu,sigma] = standardizeCols(X);
X = [ones(n,1) X];
d = d + 1;
 
% Make sure to apply the same transformation to the validation/test data
Xvalid = standardizeCols(Xvalid,mu,sigma);
Xvalid = [ones(t,1) Xvalid];
Xtest = standardizeCols(Xtest,mu,sigma);
Xtest = [ones(t2,1) Xtest];
 

% Choose network structure
nHidden = [120 60 30];
ncov=25;%卷积网络的5*5=25的kernel，变成一长条的weight
 
% Count number of parameters and initialize weights 'w'
nParams = d*nHidden(1)+25;
for h = 2:length(nHidden)
    nParams = nParams+nHidden(h-1)*nHidden(h);
end
nParams = nParams+nHidden(end)*nLabels;
w = unifrnd(-0.25,0.25,nParams,1);
 
% Train with stochastic gradient
%record all w
maxIter = 100000;
stepSize =2e-3;
J=[];
funObj = @(w,i)MLPclassificationLoss6(w,X(i,:),yExpanded(i,:),nHidden,nLabels);
iter = 1;
if mod(iter-1,round(maxIter/20)) == 0
   yhat = MLPclassificationPredict3(w,Xvalid,nHidden,nLabels);
   A= sum(yhat~=yvalid)/t;
   fprintf('Training iteration = %d, validation error = %f\n',iter-1,A);
end
i = ceil(rand*n);
[f,g] = funObj(w,i);
w = w - stepSize*g;
J=[w];
b=0.9;
h = waitbar(0,'Please wait...');

for iter = 2:maxIter
    if mod(iter-1,round(maxIter/20)) == 0
        yhat = MLPclassificationPredict3(w,Xvalid,nHidden,nLabels);
        fprintf('Training iteration = %d, validation error = %f\n',iter-1,sum(yhat~=yvalid)/t);
    end
   
    if sum(yhat~=yvalid)/t >A 
         break
    else A=sum(yhat~=yvalid)/t;
end
    i = ceil(rand*n);
    [f,g] = funObj(w,i);
    w= w - stepSize*g+b*(w-J);
    J=[w];
    waitbar((iter-1)/(maxIter-1),h)
end  
 
 
% Evaluate test error
yhat = MLPclassificationPredict3(w,Xtest,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);
pp=toc;
fprintf('time= %f\n',pp);
