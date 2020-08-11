/*Modified version of a code written originally by Mingyuan Zhou*/
#include "mex.h"

mwIndex BinarySearch(double probrnd, double *prob_cumsum, mwSize Ksize) {
    mwIndex k, kstart, kend;
    if (probrnd <=prob_cumsum[0])
        return(0);
    else {
        for (kstart=1, kend=Ksize-1; ; ) {
            if (kstart >= kend) {
                return(kend);
            }
            else {
                k = kstart+ (kend-kstart)/2;
                if (prob_cumsum[k-1] > probrnd && prob_cumsum[k]>probrnd)
                    kend = k-1;
                else if (prob_cumsum[k-1]<probrnd && prob_cumsum[k]<probrnd)
                    kstart = k+1;
                else
                    return(k);
            }
        }
    }
    return(k);
}

void Multrnd_Matrix(double *ZSDS, double *WSZS, double *Phi, double *Theta, mwIndex *ir, mwIndex *jc, double *pr, mwSize Vsize, mwSize Nsize, mwSize Ksize,  double *prob_cumsum, double mm, double *nm)
{
    
    double cum_sum, probrnd;
    mwIndex k, j, v, token, ji=0, total=0, table, counter=0;
    mwIndex starting_row_index, stopping_row_index, current_row_index;
    
    
    for (j=0;j<Nsize;j++) {
        starting_row_index = jc[j];
        stopping_row_index = jc[j+1];
        if (starting_row_index == stopping_row_index)
            continue;
        else {
            for (current_row_index =  starting_row_index; current_row_index<stopping_row_index; current_row_index++) {
                v = ir[current_row_index];
                for (cum_sum=0,k=0; k<Ksize; k++) {
                    cum_sum += Phi[v+ k*Vsize]*Theta[k];
                    prob_cumsum[k] = cum_sum;
                }
                if (pr[total]>mm) {
                    for (token=1, table=1;token< (mwIndex) mm;token++) {
                        if  (((double) rand() / RAND_MAX) <= (cum_sum/(cum_sum+ token)))
                            table++;
                    }
                    table += nm[counter];
                    counter++;
                }
                else {
                    for (token=1, table=1;token< (mwIndex) pr[total];token++) {
                        if  (((double) rand() / RAND_MAX) <= (cum_sum/(cum_sum+ token)))
                            table++;
                    }
                }
                
                for (token=0;token< table;token++) {
                    probrnd = (double) rand() / RAND_MAX *cum_sum;
                    ji++;
                    k = BinarySearch(probrnd, prob_cumsum, Ksize);
                    
                    ZSDS[k+Ksize*j]++;
                    WSZS[v+k*Vsize]++;
                }
                total++;
            }
        }
    }
}

void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    double *ZSDS, *WSZS, *Phi, *Theta, *RND;
    double  *pr, *prob_cumsum, *nm, mm;
    mwIndex *ir, *jc;
    mwIndex Vsize, Nsize, Ksize;
    mxArray *prob_cumsum_Array;
    

    pr = mxGetPr(prhs[0]);
    ir = mxGetIr(prhs[0]);
    jc = mxGetJc(prhs[0]);
    Vsize = mxGetM(prhs[0]);
    Nsize = mxGetN(prhs[0]);
    Ksize = mxGetN(prhs[1]);
    Phi = mxGetPr(prhs[1]);
    Theta = mxGetPr(prhs[2]);
    mm = mxGetScalar(prhs[3]);
    nm = mxGetPr(prhs[4]);
    
    plhs[0] = mxCreateDoubleMatrix(Ksize,Nsize,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(Vsize,Ksize,mxREAL);
    ZSDS = mxGetPr(plhs[0]);
    WSZS = mxGetPr(plhs[1]);
    
    
    prob_cumsum_Array = mxCreateDoubleMatrix(Ksize,1,mxREAL);
    prob_cumsum = mxGetPr(prob_cumsum_Array);
    
    
    Multrnd_Matrix(ZSDS, WSZS, Phi, Theta, ir, jc, pr,  Vsize, Nsize, Ksize, prob_cumsum, mm, nm);
    mxDestroyArray(prob_cumsum_Array);
}