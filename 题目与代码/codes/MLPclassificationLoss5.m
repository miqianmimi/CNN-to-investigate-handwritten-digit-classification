%改过第三，四，五，七
function [f,g] = MLPclassificationLoss5(w,X,y,nHidden,nLabels)
dropoutFraction = 0.5;
lambda=0.02;
[nInstances,nVars] = size(X);

% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1));
offset = nVars*nHidden(1);
for h = 2:length(nHidden)
hiddenWeights{h-1} = reshape(w(offset+1:offset+nHidden(h-1)*nHidden(h)),nHidden(h-1),nHidden(h));
offset = offset+nHidden(h-1)*nHidden(h);
end
outputWeights = w(offset+1:offset+nHidden(end)*nLabels);
outputWeights = reshape(outputWeights,nHidden(end),nLabels);

f = 0;
if nargout > 1
gInput = zeros(size(inputWeights));
for h = 2:length(nHidden)
gHidden{h-1} = zeros(size(hiddenWeights{h-1}));
end
gOutput = zeros(size(outputWeights));
end

%include dropout
for i = 1:nInstances
dropoutmatrix{1}=(rand(size(X( i ,:)))>0.5);
X(i,:)= X(i,:).*dropoutmatrix{1};
ip{1} = X(i,:)*inputWeights;
fp{1} = tanh(ip{1});
for h = 2:length(nHidden)
dropoutmatrix{h}=(rand(size(fp{h-1}))>0.5);
fp{h-1}= fp{h-1}.*dropoutmatrix{h};
ip{h} = fp{h-1}*hiddenWeights{h-1};
fp{h} = tanh(ip{h});
end
yhat = fp{end}*outputWeights;


%softmax
%choice the right yhat
for m=1:10
if y(m)==1
break
end
end
k=exp(yhat(m));
%compute Loss function
p=0;
for j=1:10
p=exp(yhat(j))+p;
end
probability=exp(yhat)/p;
f=f -log(k/p);

if nargout > 1
%output layer
gOutput = fp{end}'*exp(yhat)/p+lambda*(outputWeights);
gOutput(:,m) = gOutput(:,m)-fp{end}';

if length(nHidden) > 1
% Last Layer of Hidden Weights
clear backprop
clear backprop1%/p
backprop = exp(yhat)*(repmat(sech(ip{end}).^2,nLabels,1).*outputWeights')/p;
backprop1= sech(ip{end}).^2.*outputWeights(:,m)';
fp{end-1}=fp{end-1}.*dropoutmatrix{end};
gHidden{end} = gHidden{end} + fp{end-1}'*backprop-fp{end-1}'*backprop1+lambda*hiddenWeights{length(nHidden)-1};

backprop = sum(backprop,1);

% Other Hidden Layers
for h = length(nHidden)-2:-1:1

backprop = (backprop*hiddenWeights{h+1}').*sech(ip{h+1}).^2;
backprop1 = (backprop1*hiddenWeights{h+1}').*sech(ip{h+1}).^2;
fp{h}=ip{h}.*dropoutmatrix{h+1};
gHidden{h} = gHidden{h} +(fp{h}'*backprop-fp{h}'*backprop1)/p+lambda*(hiddenWeights{h});

end

% Input Weights
backprop = (backprop*hiddenWeights{1}').*sech(ip{1}).^2;
backprop1 = (backprop1*hiddenWeights{1}').*sech(ip{1}).^2;
X(i,:)=X(i,:).*dropoutmatrix{1};
gInput = gInput + (X(i,:)'*backprop- X(i,:)'*backprop1)/p+ lambda*(inputWeights);

else
%nargout = 1
X(i,:)=X(i,:).*dropoutmatrix{1};
mya= X(i,:)'*exp(yhat)*(repmat(sech(ip{end}).^2,nLabels,1).*outputWeights');
myq =X(i,:)'*p*(sech(ip{end}).^2.*outputWeights(:,m)');
gInput = gInput+(mya-myq)/p+lambda*(inputWeights);
end

end

end

% Put Gradient into vector
if nargout > 1
g = zeros(size(w));
g(1:nVars*nHidden(1)) = gInput(:);
offset = nVars*nHidden(1);
for h = 2:length(nHidden)
g(offset+1:offset+nHidden(h-1)*nHidden(h)) = gHidden{h-1};
offset = offset+nHidden(h-1)*nHidden(h);
end
g(offset+1:offset+nHidden(end)*nLabels) = gOutput(:);
end
