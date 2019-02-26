function [h,yb,xb] = coint(PriceX,PriceY)
% h adftest, yb -> alpha11, xb -> alpha21

% regression: Y ~ X + 1
reg = fitlm(PriceX,PriceY) ;
  
% residuals
err = reg.Residuals.Raw;

% price difference
dx = diff(PriceX);
dy = diff(PriceY);
err = err(2:end);

% vecm regression
vecm_y = fitlm(err,dy);        
vecm_x = fitlm(err,dx);
yb = vecm_y.Coefficients{'x1','Estimate'};
xb = vecm_x.Coefficients{'x1','Estimate'};

% unit root test
h = adftest(err,'model','AR','lags',2);

end

