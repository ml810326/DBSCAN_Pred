#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include <dirent.h>
#include <time.h>

#include<sys/types.h>
#include<sys/stat.h>
#include<fcntl.h>
#include <unistd.h>

#include "blob.h"

#define NUMBERDATA 10000

//Define Gsensor data 
typedef struct{
    float gx; //Gsensor_X data
    float gy; //Gsensor_Y data
    float gz; //Gsensor_Z data
} gsensor_d;

int main(int argc, char *argv[]) {

    FILE *fp;
    gsensor_d gdata[NUMBERDATA];
    int count = 0;

    //open model data to read
    fp = fopen("model.csv", "r");
    if(fp == NULL) {
        fprintf(stderr, "Error reading file\n");
        return 1;
    }
    //read Gsensor data and put into gdata
    while(fscanf(fp, "%f,%f,%f", &gdata[count].gx, &gdata[count].gy, &gdata[count].gz) == 3){
        count++;
    }

    fclose(fp);

    //Define the threadhold for application    
    float eps = 1.0;

    //new data    
    float new_points[3] = {atof(argv[1]), atof(argv[2]), atof(argv[3])};
    int i = 0;
    bool abnormal = true;

    //check the data distance with model
    for(i = 0; i < count; i++){
        float sum = pow((new_points[0] - gdata[i].gx), 2) + pow((new_points[1] - gdata[i].gy), 2) + pow((new_points[2] - gdata[i].gz), 2);
        sum = pow(sum, 0.5);
        printf("sqrsum = %f\n", sum);
        if(sum < eps){
            abnormal = false;
            break;
        }
        sum = 0;
    }

    //show the result
    if(abnormal){
        printf("data is abnormal\n");
    }else{
        printf("data is normal\n");
    }

    return 0;
}
