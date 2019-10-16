function output = mahalanobis(x,m,Sigma)
output = sqrt( (x-m)*Sigma*transpose(x-m));
end

