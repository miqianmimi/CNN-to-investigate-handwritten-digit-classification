%卷积神经网络的loss6函数
function [f,g] = MLPclassificationLoss6(w,X,y,nHidden,nLabels)
lambda=0.02;
[nInstances,nVars] = size(X);
%conv weights
convWeights=reshape(w(1:25),5,5);

%compute conv2
for i=1:nInstances
    convInput=reshape(X(i,2:257),16,16);
    convOutput=conv2(convInput,convWeights,'valid');%convOutput是一个12*12的矩阵,valid使得卷积出来结果对
end

%pooling(average pooling)
for i=1:nInstances
   poolingmatrix=[0.25,0.25;0.25,0.25];
   poolingOutput=conv2(convOutput,poolingmatrix,'valid');
   poolingOutput=poolingOutput([1,3,5,7,9,11],[1,3,5,7,9,11]);%把poolingOutput变成6*6的矩阵
end

%a brandnew 6*6 input=x
X=reshape(poolingOutput,1,36);
[nInstances,nVars] = size(X);


% Form Weights
inputWeights = reshape(w(26:25+nVars*nHidden(1)),nVars,nHidden(1));%前25个一定是kernel的weight
offset = nVars*nHidden(1)+25;
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


% Compute Output

for i = 1:nInstances
    ip{1} = X*inputWeights;
    fp{1} = tanh(ip{1});
    for h = 2:length(nHidden)
        ip{h} = fp{h-1}*hiddenWeights{h-1};
        fp{h} = tanh(ip{h});
    end
    yhat = fp{end}*outputWeights;
    
    relativeErr = yhat-y(i,:);
    f = f + sum(relativeErr.^2);
    
    if nargout > 1
        err = 2*relativeErr;
 
        % Output Weights
        %for c = 1:nLabels
        %   gOutput(:,c) = gOutput(:,c) + err(c)*fp{end}';
        %end
        gOutput = gOutput+(fp{end}'*err)+lambda*(outputWeights);
       
 
        if length(nHidden) > 1
            % Last Layer of Hidden Weights
            clear backprop
           backprop = err*(repmat(sech(ip{end}).^2,nLabels,1).*outputWeights');
           gHidden{end} = gHidden{end} + fp{end-1}'*backprop+lambda*hiddenWeights{length(nHidden)-1};       
           backprop = sum(backprop,1);
 
            % Other Hidden Layers
            for h = length(nHidden)-2:-1:1
                backprop = (backprop*hiddenWeights{h+1}').*sech(ip{h+1}).^2;
                gHidden{h} = gHidden{h} + fp{h}'*backprop+lambda*(hiddenWeights{h});
            end
 
            % Input Weights
            backprop = (backprop*hiddenWeights{1}').*sech(ip{1}).^2;
            gInput = gInput + X(i,:)'*backprop+ lambda*(inputWeights);
            
            %add a derivative of convweight:这边是卷积的网络的导数
            backprop=backprop*inputWeights'*1;
            backprop=reshape(backprop,6,6);
            expanded_backprop=kron(backprop,ones(2));
            gconv=Rot180(conv2(convInput,Rot180(expanded_backprop),'valid'));
            
        else
           % Input Weights   
           %  for c = 1:nLabels
           %  gInput = gInput + err(c)*X(i,:)'*(sech(ip{end}).^2.*outputWeights(:,c)');
           % end
            
               
            gInput = gInput+X(i,:)'*err*(repmat(sech(ip{end}),nLabels,1).^2*outputWeights')+lambda*(inputWeights);
            backprop=backprop*inputWeights'*1;
            backprop=reshape(backprop,6,6);
            expanded_backprop=kron(backprop,ones(2));
            gconv=conv2(convOutput,Rot180(expanded_backprop),'same');
            gconv=Rot180(conv2(convInput,gconv,'valid'));
        end
 
    end
    
end
 
% Put Gradient into vector
if nargout > 1
    g = zeros(size(w));
    g(1:25)=gconv(:);
    g(26:nVars*nHidden(1)+25) = gInput(:);
    offset = nVars*nHidden(1)+25;
    for h = 2:length(nHidden)
        g(offset+1:offset+nHidden(h-1)*nHidden(h)) = gHidden{h-1};
        offset = offset+nHidden(h-1)*nHidden(h);
    end
    g(offset+1:offset+nHidden(end)*nLabels) = gOutput(:);
end


