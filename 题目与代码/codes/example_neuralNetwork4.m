%改过2,不含第4题stop,5,8,加过第八题的,最后一层用Linear regression,加过第九问的
tic
load digits.mat
X1=X;
for i=1:5000*256
    if X1(i)~=0
       X1(i)= X1(i)+10*(-1+2*rand); 
    end
end
for i=1:5000 
    Img{i}=reshape(X1(i,:),[16,16]);
    X1=circshift(X1,1);
    X1(1,:)=0;
end
X=[X;X1];
y=[y;y];
[n,d] = size(X);
nLabels = max(y);
yExpanded = linearInd2Binary(y,nLabels);
t = size(Xvalid,1);
t2 = size(Xtest,1);
 
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
 
% Count number of parameters and initialize weights 'w'
nParams = d*nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams+nHidden(h-1)*nHidden(h);
end
nParams = nParams+nHidden(end)*nLabels;
w = unifrnd(-0.25,0.25,nParams,1);
 
% Train with stochastic gradient
%record all w
maxIter = 100000;
weights=zeros(nParams,2);
weights(:,1)=w;
b=0.9;
stepSize =2e-3;
funObj = @(w,i)MLPclassificationLoss2(w,X(i,:),yExpanded(i,:),nHidden,nLabels);
h = waitbar(0,'Please wait...');
for iter = 1:maxIter
if mod(iter-1,round(maxIter/20)) == 0
   [yhat,j] = MLPclassificationPredict2(w,Xvalid,nHidden,nLabels);
   k=zeros(size(j,2),nLabels);
   yExpanded1=linearInd2Binary(yvalid,nLabels);
for l=1:nLabels
model=leastSquares(j,yExpanded1(:,l));
k(:,l)=model.w;
end
yhat1=j*k;
[v,yhat1]=max(yhat1,[],2);
fprintf('Training iteration = %d, validation error = %f\n',iter-1, sum(yhat1~=yvalid)/t);
end

    i = ceil(rand*n);
[f,g] = funObj(w,i);
if iter ==1
      w=w-stepSize*g;
      weights(:,2)=w;
else
      w=w-stepSize*g+b*(w-weights(:,2));
      weights(:,1)=weights(:,2);
      weights(:,2)=w;
    if iter>maxIter/5
        stepSize=2e-3;
        b=0.9;
    end
    if iter>maxIter*2/5
        stepSize=2e-3;
        b=0.75;
    end
    if iter>maxIter*3/5
        stepSize=1e-3;
        b=0.5;
    end
    if iter>maxIter*4/5
        stepSize=5e-4;
        b=0.35;
    end
end
    waitbar((iter-1)/(maxIter-1),h)
end  
 
 
% Evaluate test error
[yhat,j]= MLPclassificationPredict2(w,Xtest,nHidden,nLabels);
   k=zeros(size(j,2),nLabels);
   yExpanded1=linearInd2Binary(ytest,nLabels);
for l=1:nLabels
model=leastSquares(j,yExpanded1(:,l));
k(:,l)=model.w;
end
yhat1=j*k;
[v,yhat1]=max(yhat1,[],2);
fprintf('Test error with final model = %f\n',sum(yhat1~=ytest)/t2);
pp=toc;
fprintf('time= %f\n',pp);
 
 
