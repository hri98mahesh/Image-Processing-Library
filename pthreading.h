#include <iostream>
#include <time.h>
#include <pthread.h>
#include <vector>
using namespace std;
#ifndef PTHREADING_H_INCLUDE
#define PTHREADING_H_INCLUDE

int cou=0;
int loop_size=0;  
struct threeMatrix{
    vector<vector<float> > *First;
    vector<vector<float> > *Second;
    vector<vector<float> > Answer;
  
};

void* multi(void* ac) 
{ 
    struct threeMatrix *number = (struct threeMatrix *)ac;
    vector<vector<float> > temp1_First;
    vector<vector<float> > temp1_Second;
    vector<vector<float> > temp1_Answer;
    temp1_First=*(number->First);
    temp1_Second=*(number->Second);
    temp1_Answer=(number->Answer);
    

    int row = (cou)++;
    void* ope;

        for(int i=row*loop_size/4;i<(row+1)*loop_size/4;i++){
            for (int j = 0; j < temp1_Second[0].size(); j++){   
                    for (int k = 0; k < temp1_Second.size(); k++)  
                        temp1_Answer[i][j] += (temp1_First[i][k]) * (temp1_Second[k][j]);         
            }
            number->Answer[i]=temp1_Answer[i];
        }
        

    return ope;    
}

vector<vector<float> > matrix_mult_pthread(vector<vector<float> > A,vector<vector<float> >B){
        struct threeMatrix ac;
        ac.First=&A;
        ac.Second=&B;

        vector<vector<float> > matrix_ans(A.size(),vector<float> (B[0].size()));
        ac.Answer=matrix_ans;
        pthread_t threads[A.size()];
        loop_size=A.size();
        for (int i = 0; i < 4; i++) { 
            pthread_create(&threads[i], NULL, multi, (void*)(&ac));
            
        }
        
        for (int i = 0; i < 4; i++)  
        pthread_join(threads[i], NULL);
        vector<vector<float> > output= (ac.Answer);
        cou=0;
        //display(output);
        return output;  
}

#endif