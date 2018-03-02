%�Ĺ���ʮ��������ġ�
function [y] = MLPclassificationPredict3(w,X,nHidden,nLabels)
%�Ȱѷ����5000*256���6*6�ľ���
[nInstances,nVars] = size(X);
%conv weights
convWeights=reshape(w(1:25),5,5);
%compute conv2
for i=1:nInstances%nInstance=1
    convInput=reshape(X(i,2:1:257),16,16);
    convOutput=conv2(convInput,convWeights,'valid');%convOutput��һ��12*12�ľ���,validʹ�þ�����������
end

%pooling(average pooling)%nInstance=1
for i=1:nInstances
   poolingmatrix=[0.25,0.25;0.25,0.25];
   poolingOutput=conv2(convOutput,poolingmatrix,'valid');
   poolingOutput=poolingOutput([1,3,5,7,9,11],[1,3,5,7,9,11]);%��poolingOutput���6*6�ľ���
end

%a brandnew 6*6 input=x
X=reshape(poolingOutput,1,36);
[nInstances,nVars] = size(X);
%����ninstances nvars


% Form Weights
inputWeights = reshape(w(26:nVars*nHidden(1)+25),nVars,nHidden(1));
offset = nVars*nHidden(1)+25;
for h = 2:length(nHidden)
  hiddenWeights{h-1} = reshape(w(offset+1:offset+nHidden(h-1)*nHidden(h)),nHidden(h-1),nHidden(h));
  offset = offset+nHidden(h-1)*nHidden(h);
end
outputWeights = w(offset+1:offset+nHidden(end)*nLabels);
outputWeights = reshape(outputWeights,nHidden(end),nLabels);

% Compute Output
for i = 1:nInstances
    ip{1} = X(i,:)*inputWeights;
    fp{1} = tanh(ip{1});
    for h = 2:length(nHidden)
        ip{h} = fp{h-1}*hiddenWeights{h-1};
        fp{h} = tanh(ip{h});
    end
    y(i,:) = fp{end}*outputWeights;
end
[v,y] = max(y,[],2);
%y = binary2LinearInd(y);
