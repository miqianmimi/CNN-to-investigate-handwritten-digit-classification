function [X] = expanddata(X)
for i=1:5000
    Img{i}=reshape(X(i,:),[16,16]);
    tform=affine2d([0.5 0 0; 0.5 1 0;0 0 1]);
    J = imwarp(Img{i},tform);
    J= reshape(J,[1,256]);
    X=[X;J];
end
end
