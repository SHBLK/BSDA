function [y_out,err]=get_NB_error_OBC(X,obc_likelihood,p,u_mask,l_mask,y,labs,y_true,prior_ratio)

%OBC Labeling and Error Calculation

d_idx = length(X);
y_out = y{d_idx};
N_u = sum(u_mask{d_idx});
err = zeros(length(labs{d_idx}),4);%TP,FP,TN,FN
if size(prior_ratio{d_idx},1)==1
   like = obc_likelihood + log(prior_ratio{d_idx})'; 
else
   like = obc_likelihood + log(prior_ratio{d_idx}); 
end
[m,I]=max(like);
y_out(u_mask{d_idx}) = labs{d_idx}(I);
        
for i_cnt=1:length(labs{d_idx})
            TAP = y_true{d_idx}(u_mask{d_idx})==labs{d_idx}(i_cnt);
            PP = y_out(u_mask{d_idx})==labs{d_idx}(i_cnt);
            TP=sum(bitand(TAP,PP));
            FP = max(sum(PP) - TP,0);
            TAN = 1 - TAP;
            PN = 1 - PP;
            TN=sum(bitand(TAN,PN));
            FN = max(sum(PN) - TN,0);
            err(i_cnt,:) = [TP,FP,TN,FN];
end
    

end
