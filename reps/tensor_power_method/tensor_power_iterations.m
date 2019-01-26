function [muhat, what] = tensor_power_iterations(X,params)
%Tensor Power Iterations (for GMM - you can set sig = 0 for topic model)
%Inputs:
%X, LxN data matrix
%params, The parameter struct. It has the ollowing fields:
%K - Number of hidden components
%E - Number of power iterations
%v - Variance of the observation model, see the tensor factorization paper for details
%display - Toggle to display iterations, if 1 code shows the iterations
%
%Outputs:
%muhat, LxK estimated means matrix
%what, 1xK estimated mixing proportions
%Cem Subakan, UIUC 2014

L = size(X,1);
N = size(X,2);
K = params.K;
E = params.E;
v = params.v;


[U, V] = eigs( (X*X'/N) - v*eye(L) ,K);  %Users can write their own matrix power updates if this line bothers them

W = U * sqrt(inv(V));

u = randn(K);
u = bsxfun(@rdivide,u,sqrt(sum(u.^2)));

m = mean(X,2);

temp = W' * X;
for e = 1 : E
    
    for k = 1: K
        temp2 = (u(:,k)' * W' * X);
        
        temp3 = sum( (u(:,k)'* W').^2 );
        
        eval(k) = mean(temp2.^3,2) - 3*v*( u(:,k)'* W'*m * temp3 );
        
        u(:,k) = mean(bsxfun(@times,temp,temp2.^2),2) - v*( (W'*m * temp3) + (2*u(:,k)'* W'*m *(W'*W*u(:,k))) ) ; %power iteration
        
        
    end
    [u,~] = qr(u); %orthogonalize
    
    if params.display
        disp(u)
        disp(eval)
        pause(0.1)
    end
    
end

muhat = pinv(W')*u*diag(eval);
what = (1./eval).^2;