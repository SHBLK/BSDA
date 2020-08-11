function [output,resCell]=...
    OBSDA(X,y,y_true,prior_ratio,eta,flag_sample_phi,K,burnin,collection,CollectionStep)
%X: A cell array of matrices containing the counts for domains (target domain last), rows corresponding to genes and columns corresponding to samples.
%y: A cell array of vectors containing the labels for domains (target domain last). 0 indicates unlabeled data.
%y_true: A cell array of vectors containing the labels for all data in the domains (target domain last).
%eta: Dirichlet prior parameter: J*K or scalar
%prior ratio: Class probabilities for the domains. (sorted with increasing label number indicator)
%flag_sample_phi: 1: Sample phi   0: Keep phi fixed at its prior mean
%K: Number of factors (default 32)
%burnin: Gibbs chain burin, start collecting after burin iterations
%collection: Gibbs chain iterations to collect sample after burnin
%CollectionStep: Thinning, collect a sample every CollectionStep iterations
model = 'OBSDA';
sampler = 'Gibbs';
if ~exist('eta','var')
    eta=0.05;
end
if ~exist('K','var')
    K=32;
end
if ~exist('burnin','var')
    burnin=50;
end
if ~exist('collection','var')
    collection=250;
end
if ~exist('CollectionStep','var')
    CollectionStep=5;
end

if ~exist('flag_sample_phi','var')
    flag_sample_phi = true;
end


n_d = length(X);
all_labs = [];
for d_idx=1:n_d
    unlabeled_mask{d_idx} = y{d_idx}==0;
    labeled_mask{d_idx} = y{d_idx}~=0;
    y_temp=y{d_idx}(y{d_idx}~=0);
    labs{d_idx}=unique(y_temp);
    [V,N{d_idx}] = size(X{d_idx});
    P{d_idx}=V;
    %Initialization for speed
    for i_cnt=1:length(labs{d_idx})
        Theta{d_idx}{i_cnt} = zeros(K,1)+1/K;
        PhiTheta{d_idx}{i_cnt}=[];
        XtrainSparse{d_idx}{i_cnt}=[];
        ii{d_idx}{i_cnt}=[];
        rr{d_idx}{i_cnt}=[];
        nm{d_idx}{i_cnt}=[];
        WSZS{d_idx}{i_cnt}=[];
        ZSDS{d_idx}{i_cnt}=[];
    end
    all_labs = [all_labs;unique(y_temp)];
end
all_labs=unique(all_labs);
all_labs_count = length(unique(all_labs));

if length(eta)==1
    Phi = rand(V,K);
    Phi = bsxfun(@rdivide,Phi,sum(Phi,1));
else
    if flag_sample_phi
        Phi = drchrnd(eta);
    else
        Phi = eta./sum(eta,1);
    end

end

c_0=1;gamma0=1; b_k= 50/K*ones(K,1);
q_z=2*ones(1,n_d);
for d_idx=1:n_d
    p_i{d_idx}=0.5*ones(1,N{d_idx});
    u_k{d_idx} = 50/K*ones(K,1);
    for i_cnt=1:length(labs{d_idx})%pi
        p_i_class{d_idx}{i_cnt}=0.5*ones(1,sum(unlabeled_mask{d_idx}));
    end
end
v_l = ones(1,all_labs_count);

g0=1e-2; h0=1e-2;
alpha0=.1; beta0=.1;
w0=.01;u0=.01;
e0=1.0;f0=1.0;
a0=1.0;d0=1.0;


mm = 1500;%Approximate CRT for larger than this value

fprintf('\n');

%Initialization for speed
p_hat = zeros(n_d,all_labs_count);

sumlogpi_tilde_tilde=zeros(n_d,1);
L_k=zeros(K,n_d,all_labs_count);%This is tilde n
for d_idx=1:n_d
   p_tilde_i{d_idx} = zeros(1,N{d_idx});
   p_tilde_tilde_i{d_idx} = zeros(1,N{d_idx});
end
err=[];
curr_err=[];
resCell=[];
rescell_cnt = 1;
all_err=zeros(1,burnin + collection);

%

tic
for iter=1:burnin + collection
    iter 
    if strcmp(model, 'OBSDA')
        switch sampler
            case {'Gibbs'}
                for d_idx = 1:n_d
                        for i_cnt=1:length(labs{d_idx})
                            i=labs{d_idx}(i_cnt);
                            label_mask = y{d_idx}==labs{d_idx}(i_cnt);
                            N_zl=length(label_mask);
                            PhiTheta{d_idx}{i_cnt} = Phi * repmat(Theta{d_idx}{i_cnt},1,N_zl);                        
                            XtrainSparse{d_idx}{i_cnt}= sparse(X{d_idx}(:,label_mask));
                            ii{d_idx}{i_cnt} = find(XtrainSparse{d_idx}{i_cnt}>mm);
                            rr{d_idx}{i_cnt} = PhiTheta{d_idx}{i_cnt}(ii{d_idx}{i_cnt});
                            nm{d_idx}{i_cnt} = poissrnd(rr{d_idx}{i_cnt}.*(psi(XtrainSparse{d_idx}{i_cnt}(ii{d_idx}{i_cnt})+rr{d_idx}{i_cnt})-psi(mm+rr{d_idx}{i_cnt})));
                            tic
                            [ZSDS{d_idx}{i_cnt},WSZS{d_idx}{i_cnt}] = CRT_Multrnd_Matrix_v2_obc(XtrainSparse{d_idx}{i_cnt},Phi,Theta{d_idx}{i_cnt}, mm, nm{d_idx}{i_cnt});
                            %ZSDS is K*N, WSZS is J*K
                            toc
                            for k=1:K
                                L_k(k, d_idx,i) = CRT_sum_mex(ZSDS{d_idx}{i_cnt}(k,:),u_k{d_idx}(k));
                            end
                            p_i{d_idx}(label_mask) = betarnd(g0 + sum(X{d_idx}(:,label_mask),1),h0+sum(Theta{d_idx}{i_cnt},1));
                            p_tilde_i{d_idx}(label_mask) = -log(max(1-p_i{d_idx}(label_mask),realmin));           
                            p_i_class{d_idx}{i_cnt} = betarnd(g0 + sum(X{d_idx}(:,unlabeled_mask{d_idx}),1),h0+sum(Theta{d_idx}{i_cnt},1));
                            p_tilde_tilde_i{d_idx}(label_mask) =  p_tilde_i{d_idx}(label_mask)./(v_l(i)+ p_tilde_i{d_idx}(label_mask));
                            p_hat(d_idx,i) = -sum(log(max(1-p_tilde_tilde_i{d_idx}(label_mask),realmin)));
                            p_hat(d_idx,i) = p_hat(d_idx,i)./(q_z(d_idx) + p_hat(d_idx,i));
                        end
                    sumlogpi_tilde_tilde(d_idx) = sum(log(max(1-p_tilde_tilde_i{d_idx},realmin)));

                end
                                
                sumlogpi_hat = sum(sum(log(max(1-p_hat,realmin))));
                p_hat_hat = -sum(log(max(1-p_hat,realmin)),1);
                p_hat_hat = p_hat_hat./(c_0+p_hat_hat);
                sumlogpi_hat_hat = sum(log(max(1-p_hat_hat,realmin)));
                for d_idx = 1:n_d
                    
                    u_k{d_idx} = gamrnd(b_k + sum(L_k(:,d_idx,:),3), 1./(q_z(d_idx) -  sumlogpi_tilde_tilde(d_idx)));
                
                end
                
                L_tilde_k=zeros(K,all_labs_count);
                for k=1:K
                    for i=1:all_labs_count
                        if sum(L_k(k,:,i))~=0
                            L_tilde_k(k,i) = L_tilde_k(k,i) + CRT_sum_mex(L_k(k,:,i),b_k(k));
                        end
                    end
                end
                

                
                L_tilde_tilde_k=zeros(1,all_labs_count);
                
                for i=1:all_labs_count
                        if sum(L_tilde_k(:,i))~=0
                            L_tilde_tilde_k(i) = L_tilde_tilde_k(i) + CRT_sum_mex(L_tilde_k(:,i),gamma0/K);
                        end
                end
                
                
                c_0 = randg(a0 + gamma0)/(d0+sum(b_k));
                gamma0 = gamrnd(alpha0 + sum(L_tilde_tilde_k),1./(beta0 - sumlogpi_hat_hat));
                b_k = randg(sum(L_tilde_k,2)+gamma0/K)./(-sumlogpi_hat+ c_0);
                
                update_phi=zeros(V,K);
                for d_idx = 1:n_d
                    
                    for i_cnt=1:length(labs{d_idx})
                          i=labs{d_idx}(i_cnt);
                          label_mask = y{d_idx}==labs{d_idx}(i_cnt);
                
                          Theta{d_idx}{i_cnt} = bsxfun(@rdivide,randg(sum(ZSDS{d_idx}{i_cnt},2) + u_k{d_idx}), v_l(i)+sum(p_tilde_i{d_idx}(label_mask)));
                    
                    end
                    update_phi = update_phi + sum(cat(3,WSZS{d_idx}{:}),3);
                end
                if flag_sample_phi
                    Phi = drchrnd(bsxfun(@plus,update_phi, eta));
                end
                for i=1:all_labs_count
                    sum_Theta_l = 0;
                    for d_idx = 1:n_d
                            label_index= i==labs{d_idx};
                            sum_u_zkl = 0;
                            if sum(label_index)>0
                                sum_Theta_l = sum_Theta_l + sum(Theta{d_idx}{label_index});
                                sum_u_zkl = sum_u_zkl + sum(u_k{d_idx});
                            end
                        
                    end
                    v_l(i) = randg(e0+sum_u_zkl)./(f0+sum_Theta_l);
                end
                
                sum_u_k = zeros(1,n_d);
                for d_idx=1:n_d
                    sum_u_k(d_idx) = sum(u_k{d_idx});
                end
                q_z = gamrnd(w0+sum(b_k),1./(u0+sum_u_k));
                
                [y_pred,mm_l,err,obc_prob_tmp]=get_NB_likelihood_OBC_pi(X, Phi, Theta, p_i,p_i_class,unlabeled_mask,labeled_mask,y,labs,y_true,prior_ratio,0);
                curr_err = (err{end}(1,4) + err{end}(2,4)) ./sum(err{end}(1,:))
                all_err(iter) = curr_err;
        end
        
    end    
    
    if iter > burnin && mod(iter,CollectionStep)==0
        resCell{rescell_cnt}.obc_prob = obc_prob_tmp;
        resCell{rescell_cnt}.err = err;
        obc_prob{rescell_cnt} = obc_prob_tmp{n_d};
        rescell_cnt = rescell_cnt + 1;
    end

    if mod(iter,100)==0
        text = sprintf('Train Iter: %d',iter); fprintf(text, iter);
        text = sprintf('Current Error: %d',curr_err); fprintf(text, curr_err);
    end
end
[MM,II]=min(all_err);
text = sprintf('Lowest Observed Error at: %d',II); fprintf(text, II);
disp(' ')
text = sprintf('Lowest Observed Error was: %d',MM); fprintf(text, MM);
disp(' ')
output.obc_ave_lik = logsumexp(cat(3,obc_prob{:}),3) - log(rescell_cnt-1);
[obc_y_out,obc_err]=get_NB_error_OBC(X,output.obc_ave_lik, p_i,unlabeled_mask,labeled_mask,y,labs,y_true,prior_ratio);
output.obc_error = obc_err;
output.obc_y_out = obc_y_out;
curr_err = (obc_err(1,4) + obc_err(2,4)) ./sum(obc_err(1,:));
text = sprintf('OBC Error was: %d',curr_err); fprintf(text, curr_err);
end
