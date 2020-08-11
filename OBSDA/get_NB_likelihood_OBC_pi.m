function [y_out,like_all,err,like_tmp]=get_NB_likelihood_OBC_pi(X, phi, theta,p,p_class,u_mask,l_mask,y,labs,y_true,prior_ratio,flag_multi)

%Labeling part & likelihood calculation
y_out = y;
like_all = zeros(length(X),1);
err=cell(1,length(X));
like_tmp = [];
like = [];
for d_idx = 1:length(X)
    N_u = sum(u_mask{d_idx});
    err{d_idx} = zeros(length(labs{d_idx}),4);%TP,FP,TN,FN
    like_tmp{d_idx} = zeros(length(labs{d_idx}),N_u);
    like{d_idx} = zeros(length(labs{d_idx}),N_u);
    like_all_tmp = zeros(length(labs{d_idx}),1);
    for i_cnt=1:length(labs{d_idx})
        R{d_idx}{i_cnt} = max(phi * theta{d_idx}{i_cnt}, realmin);
        if N_u>0
            
            %[v, n]=size(R{i});
            log_numerator = gammaln(R{d_idx}{i_cnt}+X{d_idx}(:,u_mask{d_idx})) + X{d_idx}(:,u_mask{d_idx}) .* log(p_class{d_idx}{i_cnt}) + R{d_idx}{i_cnt} .* log(1-p_class{d_idx}{i_cnt});
            log_denominator = gammaln(X{d_idx}(:,u_mask{d_idx})+1) + gammaln(R{d_idx}{i_cnt});
            like_tmp{d_idx}(i_cnt,:) = sum(log_numerator - log_denominator,1);
        end
         
        %likelihood calculation
        label_mask = y_true{d_idx}==labs{d_idx}(i_cnt);
        log_numerator_all = gammaln(R{d_idx}{i_cnt}+X{d_idx}(:,label_mask)) + X{d_idx}(:,label_mask) .* log(p{d_idx}(label_mask)) + R{d_idx}{i_cnt} .* log(1-p{d_idx}(label_mask));
        log_denominator_all = gammaln(X{d_idx}(:,label_mask)+1) + gammaln(R{d_idx}{i_cnt});
        like_all_tmp(i_cnt) = mean(sum(log_numerator_all - log_denominator_all,1));
    end
    
    like_all(d_idx) = mean(like_all_tmp);
    
    
    if N_u>0
        
        if size(prior_ratio{d_idx},1)==1
          like{d_idx} = like_tmp{d_idx} + log(prior_ratio{d_idx})'; 
        else
          like{d_idx} = like_tmp{d_idx} + log(prior_ratio{d_idx}); 
        end
        
        if flag_multi
            mrnd_p=exp(like{d_idx}-logsumexp(like{d_idx},1))';
            mrnd_p(:,end)=1-sum(mrnd_p(:,1:end-1),2);
            mrnd_realiz=mnrnd(1,mrnd_p);
            [m,I]=max(mrnd_realiz,[],2);
            y_out{d_idx}(u_mask{d_idx}) = labs{d_idx}(I);
        else
            [m,I]=max(like{d_idx});
            y_out{d_idx}(u_mask{d_idx}) = labs{d_idx}(I);        
        end

        
        for i_cnt=1:length(labs{d_idx})
            TAP = y_true{d_idx}(u_mask{d_idx})==labs{d_idx}(i_cnt);
            PP = y_out{d_idx}(u_mask{d_idx})==labs{d_idx}(i_cnt);
            TP=sum(bitand(TAP,PP));
            FP = max(sum(PP) - TP,0);
            TAN = 1 - TAP;
            PN = 1 - PP;
            TN=sum(bitand(TAN,PN));
            FN = max(sum(PN) - TN,0);
            err{d_idx}(i_cnt,:) = [TP,FP,TN,FN];
        end
    
    end
end

end
