%Sample run file for the lung-lung experiment. The main file gets the
%following inputs:
%X: A cell array of matrices containing the counts for domains (target domain last), rows corresponding to genes and columns corresponding to samples.
%y: A cell array of vectors containing the labels for domains (target domain last). 0 indicates unlabeled data.
%y_true: A cell array of vectors containing the labels for all data in the domains (target
%domain last).
%prior ratio: Class probabilities for the domains. (sorted with increasing label number indicator)


logfc = readmatrix('/DATA/Lung_logfc_1.csv');
n_t_0 = 162;
n_t_1 = 240;
n_s_0 = 576;
n_s_1 = 552;
percent_train = 0.05;
percent_source = 0.1;
count_data = readmatrix('/DATA/Lung_allexpression.csv');
[logvals,sorted_index] = sort(logfc(:,2),'descend');
num_feats = 500;
skip_feats=5;
feats_index=sorted_index(1:skip_feats:skip_feats*num_feats);

load('/DATA/indices_lung_1.mat')
luad_1 = count_data(feats_index,3:164);
luad_1_train = luad_1(:,ind_t1(1:floor(percent_train*n_t_0)));
luad_1_test = luad_1(:,ind_t1(floor(percent_train*n_t_0) + 1:end));

lusc_1 = count_data(feats_index,165:404);
lusc_1_train = lusc_1(:,ind_t2(1:floor(percent_train*n_t_1)));
lusc_1_test = lusc_1(:,ind_t2(floor(percent_train*n_t_1) + 1:end));

luad_2_all = count_data(feats_index,405:980);
luad_2 = luad_2_all(:,ind_s1(1:floor(percent_source*n_s_0)));
lusc_2_all = count_data(feats_index,981:end);
lusc_2 = lusc_2_all(:,ind_s2(1:floor(percent_source*n_s_1)));

X{1} = [luad_2,lusc_2];%Source
X{2} = [luad_1_train,luad_1_test,lusc_1_train,lusc_1_test];%Target
y_true{1}=[ones(size(luad_2,2),1);2.*ones(size(lusc_2,2),1)];%Source
y_true{2}=[ones(size(luad_1,2),1);2.*ones(size(lusc_1,2),1)];%Target
y{1}=[ones(size(luad_2,2),1);2.*ones(size(lusc_2,2),1)];%Source
y{2}=[ones(size(luad_1_train,2),1);zeros(size(luad_1_test,2),1);2.*ones(size(lusc_1_train,2),1);zeros(size(lusc_1_test,2),1)];%Target
prior_ratio{1} = [n_s_0./(n_s_0+n_s_1),n_s_1./(n_s_0+n_s_1)];%Source
prior_ratio{2} = [n_t_0./(n_t_0+n_t_1),n_t_1./(n_t_0+n_t_1)];%Target

[output,resCell]=...
    OBSDA(X,y,y_true,prior_ratio);